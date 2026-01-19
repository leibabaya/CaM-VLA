import datetime
import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from dateutil import tz
from einops import rearrange
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from mgca.datasets.data_module import DataModule
from mgca.datasets.pretrain_dataset import (MultimodalPretrainingDataset,
                                            multimodal_collate_fn)
from mgca.datasets.transforms import DataTransforms
from mgca.models.backbones.encoder import BertEncoder, ImageEncoder
from torchmetrics import Accuracy, F1Score, Precision, Recall, AUROC
import yaml
from mgca.utils.eval_show import SimpleMetricsTable,DataCollector
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class MGCA(LightningModule):
    '''Pytorch lightning implementation of MGCA'''

    def __init__(self,
                 # img_encoder: str = "vit_base",
                 img_encoder: str = "resnet_50",
                 freeze_bert: bool = True,
                 emb_dim: int = 256,
                 learning_rate: float = None,
                 momentum: float = 0.9,
                 weight_decay: float = 0.05,
                 batch_size: int = 64,
                 num_workers: int = 8,
                 num_classes: int = 21,
                 semantic_temperature: float = 10.0,
                 tau_vf_global: float = 0.5,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.semantic_temperature = semantic_temperature
        self.tau_vf_global = tau_vf_global

        # init encoders
        self.img_encoder_q = ImageEncoder(
            model_name=img_encoder, output_dim=self.hparams.emb_dim)
        self.text_encoder_q = BertEncoder(
            output_dim=self.hparams.emb_dim, freeze_bert=freeze_bert)

        self.vf_adapter = nn.Sequential(
            nn.Conv2d(self.hparams.emb_dim, self.hparams.emb_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.hparams.emb_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hparams.emb_dim, self.hparams.emb_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.hparams.emb_dim)
        )

        self.class_names = ["ACTINO", "ADC-ECC", "ADC-EMC", "AGC-ECC-NOS", "AGC-EMC-NOS", "AGC-FN", "ASC-H", "ASC-US",
                            "Atrophy", "CC", "ECC", "EMC", "FUNGI", "HSIL", "HSV", "LSIL", "MPC", "Normal", "RPC",
                            "SCC", "TRI"]

        metric_args = {
            "task": "multiclass",
            "num_classes": num_classes,  # 从超参数获取
            "average": 'macro'
        }

        metric_args_micro = {
            "task": "multiclass",
            "num_classes": num_classes,
            "average": 'micro'  # 总样本预测正确数量
        }

        self.train_acc = Accuracy(**metric_args_micro, top_k=1)
        self.train_f1 = F1Score(**metric_args)
        self.train_recall = Recall(**metric_args)
        self.train_precision = Precision(**metric_args)
        self.train_auroc = AUROC(**metric_args)

        self.val_acc = Accuracy(**metric_args_micro, top_k=1)
        self.val_f1 = F1Score(**metric_args)
        self.val_recall = Recall(**metric_args)
        self.val_precision = Precision(**metric_args)
        self.val_auroc = AUROC(**metric_args)

        self.test_acc = Accuracy(**metric_args_micro, top_k=1)

        self.data_collector = DataCollector()
        self.final_test_results = {}

    def forward(self, batch, batch_idx, split="train"):
        '''Forward step of our method'''

        # Forward of query image encoder
        img_feat_q, patch_feat_q = self.img_encoder_q(
            batch["imgs"])
        unnorm_patch_emb_q = self.img_encoder_q.local_embed(patch_feat_q) # [64, 361, 256]
        norm_patch_emb_q = F.normalize(unnorm_patch_emb_q, dim=-1)
        unnorm_img_emb_q = self.img_encoder_q.global_embed(img_feat_q) # [64, 256]
        norm_img_emb_q = F.normalize(unnorm_img_emb_q, dim=-1)


        if not hasattr(self, 'all_class_features'):
            print("正在提取所有类别的文本特征...")
            was_training = self.text_encoder_q.training
            self.text_encoder_q.eval()
            if "tokenized_descriptions" in batch:
                with torch.no_grad():
                    # --- Part A: 粗粒度文本 (用于分类) ---
                    global_tokens = batch["tokenized_descriptions"]
                    device_tokens = {
                        "input_ids": global_tokens["input_ids"].to(self.device),
                        "attention_mask": global_tokens["attention_mask"].to(self.device),
                        "token_type_ids": global_tokens["token_type_ids"].to(self.device)
                    }
                    report_feat_q, word_feat_q, word_attn_q, sents = self.text_encoder_q(
                        device_tokens["input_ids"],
                        device_tokens["attention_mask"],
                        device_tokens["token_type_ids"]
                    )
                    self.all_class_features = report_feat_q.detach()
                    self.all_class_word_features = word_feat_q.detach()
                    print(f"-> 粗粒度特征已缓存: {self.all_class_features.shape}")

        # 准备粗粒度特征 (For Classification Loss)
        cache_report_feat_q = self.all_class_features # [21, 768]
        cache_word_feat_q = self.all_class_word_features # [21, 256, 768]
        cache_unnorm_word_emb_q = self.text_encoder_q.local_embed(cache_word_feat_q) # [21, 256, 256]
        cache_unnorm_report_emb_q = self.text_encoder_q.global_embed(cache_report_feat_q) # [21, 256]

        ########### Classification Task ################
        labels_cls = batch["labels"]
        text_feat_class = F.normalize(cache_unnorm_report_emb_q, dim=-1)
        logits = torch.matmul(norm_img_emb_q, text_feat_class.T) * self.semantic_temperature
        loss_cls = F.cross_entropy(logits, labels_cls)

        ########### Visual-Fine-grained (VF) Alignment Task ################
        B, L, Dim = unnorm_patch_emb_q.shape
        H = W = int(L ** 0.5)
        feat_image = unnorm_patch_emb_q.transpose(1, 2).view(B, Dim, H, W)
        adapted_feat = self.vf_adapter(feat_image)
        patch_feat_vf = adapted_feat.flatten(2).transpose(1, 2)
        norm_patch_emb_vf = F.normalize(patch_feat_vf, dim=-1)
        pooled_adapted_feat = F.adaptive_avg_pool2d(adapted_feat, (1, 1)).squeeze(-1).squeeze(-1)
        norm_V_global_adapted = F.normalize(pooled_adapted_feat, dim=-1)

        norm_T_global_cls = F.normalize(cache_unnorm_report_emb_q[labels_cls], dim=-1)  # [B, D]
        norm_T_local_cls = F.normalize(cache_unnorm_word_emb_q[labels_cls], dim=-1)  # [B, L_word, D]

        # --- Text-to-Vision (T2V) Alignment ---
        T_Q_global = norm_T_global_cls.unsqueeze(1)
        V_K = norm_patch_emb_vf
        V_V = norm_patch_emb_vf

        attn_scores_vf = torch.matmul(T_Q_global, V_K.transpose(-2, -1))
        scale_vf = T_Q_global.size(-1) ** 0.5
        attn_weights_vf = F.softmax(attn_scores_vf / scale_vf, dim=-1)
        V_gen_driven = torch.matmul(attn_weights_vf, V_V)
        norm_V_global_driven = F.normalize(V_gen_driven.squeeze(1), dim=-1)
        V_Q_final = norm_V_global_driven  # Query: 文本驱动的全局视觉特征
        T_K_final = norm_T_global_cls  # Key: 图像类别的全局文本特征

        sim_vf = torch.matmul(V_Q_final, T_K_final.T)
        sim_vf_scaled = sim_vf / self.tau_vf_global

        M_cls = (labels_cls.unsqueeze(0) == labels_cls.unsqueeze(1)).float().detach()

        # V -> T 损失
        log_sim = F.log_softmax(sim_vf_scaled, dim=1)
        exp_log_sim_pos = torch.exp(log_sim) * M_cls
        loss_vf_multi = -torch.clamp(torch.sum(exp_log_sim_pos, dim=1), min=1e-12).log().mean()
        # T -> V 损失 (对称)
        log_sim_T = F.log_softmax(sim_vf_scaled.T, dim=1)
        exp_log_sim_pos_T = torch.exp(log_sim_T) * M_cls.T
        loss_vf_multi_T = -torch.clamp(torch.sum(exp_log_sim_pos_T, dim=1), min=1e-12).log().mean()

        # 最终 VF 损失
        loss_t2v = (loss_vf_multi + loss_vf_multi_T) / 2.0

        # --- Vision-to-Text (V2T) Alignment ---
        V_Q_adapted = norm_V_global_adapted.unsqueeze(1)
        P_K = norm_T_local_cls
        P_V = norm_T_local_cls

        attn_scores_vis = torch.matmul(V_Q_adapted, P_K.transpose(-2, -1))  # [B, 1, L]
        scale_vis = V_Q_adapted.size(-1) ** 0.5
        attn_weights_vis = F.softmax(attn_scores_vis / scale_vis, dim=-1)
        P_distilled = torch.matmul(attn_weights_vis, P_V)
        norm_P_distilled = F.normalize(P_distilled.squeeze(1), dim=-1)  # [B, D]
        V_Q_contrast = norm_V_global_adapted  # [B, D] (Query for contrast)
        P_K_contrast = norm_P_distilled  # [B, D] (Key for contrast)

        sim_vis = torch.matmul(V_Q_contrast, P_K_contrast.T)
        sim_vis_scaled = sim_vis / self.tau_vf_global

        # V -> P (Visual View -> Patch View) 损失
        log_sim_vis = F.log_softmax(sim_vis_scaled, dim=1)
        exp_log_sim_pos_vis = torch.exp(log_sim_vis) * M_cls
        loss_vis_multi = -torch.clamp(torch.sum(exp_log_sim_pos_vis, dim=1), min=1e-12).log().mean()
        # P -> V (对称损失)
        log_sim_vis_T = F.log_softmax(sim_vis_scaled.T, dim=1)
        exp_log_sim_pos_vis_T = torch.exp(log_sim_vis_T) * M_cls.T
        loss_vis_multi_T = -torch.clamp(torch.sum(exp_log_sim_pos_vis_T, dim=1), min=1e-12).log().mean()

        loss_v2t = (loss_vis_multi + loss_vis_multi_T) / 2.0

        loss_vf = (loss_t2v + loss_v2t) / 2.0

        return loss_cls, logits, labels_cls, loss_vf, loss_t2v, loss_v2t

    def training_step(self, batch, batch_idx):
        loss_cls, logits, labels_cls, loss_vf, loss_t2v, loss_v2t = self(
            batch, batch_idx, "train")
        loss = self.hparams.lambda_cls * loss_cls + self.hparams.lambda_vf * loss_vf

        preds = F.softmax(logits, dim=-1)

        self.train_acc(preds, labels_cls)
        self.train_f1.update(preds, labels_cls)
        self.train_recall.update(preds, labels_cls)
        self.train_precision.update(preds, labels_cls)
        self.train_auroc.update(preds, labels_cls)

        self.log("train_loss", loss, on_epoch=True,  prog_bar=True)
        self.log("train_loss_cls", self.hparams.lambda_cls * loss_cls, on_epoch=True,  prog_bar=True)
        self.log("train_loss_vf", self.hparams.lambda_vf * loss_vf, on_epoch=True,  prog_bar=True)
        self.log("train_t2v", loss_t2v, on_epoch=True,  prog_bar=True)
        self.log("train_v2t", loss_v2t, on_epoch=True,  prog_bar=True)
        self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True)
        self.log("train_f1", self.train_f1, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss_cls, logits, labels_cls, loss_vf, loss_t2v, loss_v2t = self(
            batch, batch_idx, "valid")

        loss = self.hparams.lambda_cls * loss_cls + self.hparams.lambda_vf * loss_vf

        preds = F.softmax(logits, dim=-1)

        self.val_acc.update(preds, labels_cls)
        self.val_f1.update(preds, labels_cls)
        self.val_recall.update(preds, labels_cls)
        self.val_precision.update(preds, labels_cls)
        self.val_auroc.update(preds, labels_cls)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_loss_vf", self.hparams.lambda_vf * loss_vf, on_epoch=True, prog_bar=True)
        self.log("val_t2v", loss_t2v, on_epoch=True, prog_bar=True)
        self.log("val_v2t", loss_v2t, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss_cls, logits, labels_cls, loss_vf, loss_t2v, loss_v2t = self(
            batch, batch_idx, "test")

        loss = self.hparams.lambda_cls * loss_cls + self.hparams.lambda_vf * loss_vf

        preds = F.softmax(logits, dim=-1)

        self.test_acc.update(preds, labels_cls)
        self.data_collector.update(preds, labels_cls)

        self.log("test_acc", self.test_acc, on_epoch=True, prog_bar=False)
        self.log("test_loss_cls", self.hparams.lambda_cls * loss_cls, on_epoch=True, prog_bar=True)
        self.log("test_loss_vf", self.hparams.lambda_vf * loss_vf, on_epoch=True, prog_bar=True)
        self.log("test_t2v", loss_t2v, on_epoch=True, prog_bar=True)
        self.log("test_v2t", loss_v2t, on_epoch=True, prog_bar=True)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def on_test_epoch_end(self):
        all_preds, all_targets = self.data_collector.compute()

        metrics_table = SimpleMetricsTable([all_targets], [all_preds], self.class_names)
        table_str, auc = metrics_table.get_table_and_auc()
        print(table_str)

        self.final_test_results = {
            'metrics_table': table_str,
            'macro_auc': float(auc),
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            self.hparams.learning_rate,
            betas=(self.hparams.momentum, 0.999),
            weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.training_steps,
            cycle_mult=1.0,
            max_lr=self.hparams.learning_rate,
            min_lr=1e-8,
            warmup_steps=int(self.training_steps * 0.4)
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # parser.add_argument("--img_encoder", type=str, default="vit_base")
        parser.add_argument("--img_encoder", type=str, default="resnet_50")
        parser.add_argument("--emb_dim", type=int,
                            default=256, help="128, 256")
        parser.add_argument("--num_workers", type=int, default=10)
        parser.add_argument("--num_classes", type=int, default=21)
        parser.add_argument("--softmax_temperature", type=float, default=0.07)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--weight_decay", type=float, default=0.05)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--experiment_name", type=str, default="")
        parser.add_argument("--lambda_cls", type=float, default=1.)
        parser.add_argument("--lambda_vf", type=float, default=1.)
        parser.add_argument("--semantic_temperature", type=float, default=10.0)
        parser.add_argument("--tau_vf_global", type=float, default=0.1)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--data_pct", type=float, default=1.)
        return parser

    @staticmethod
    def _use_ddp_or_dpp2(trainer: Trainer) -> bool:
        if trainer and trainer.strategy:
            return isinstance(trainer.strategy, DDPStrategy)
        return torch.distributed.is_initialized()

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, trainer.num_devices)
        effective_batch_size = trainer.accumulate_grad_batches * num_devices

        return (dataset_size // effective_batch_size) * trainer.max_epochs


@torch.no_grad()
def concat_all_gather(tensor):
    '''
    Performs all_gather operation on the provided tensors
    '''
    tensors_gather = [torch.ones_like(tensor) for _ in range(
        torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output

def cli_main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = ArgumentParser()
    # trainer args
    parser = Trainer.add_argparse_args(parser)
    # model args
    parser = MGCA.add_model_specific_args(parser)
    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')

    args.deterministic = True
    args.max_epochs = 60

    # seed
    seed_everything(args.seed)

    datamodule = DataModule(MultimodalPretrainingDataset, multimodal_collate_fn,
                            DataTransforms, args.data_pct,
                            args.batch_size, args.num_workers)

    # Add load from checkpoint
    model = MGCA(**args.__dict__)

    experiment_name = (args.experiment_name + "&带缓冲层&VF对齐&语义分类&60epoch&structured描述")

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(
        BASE_DIR, f"../../../data/ckpts/semantic_vf_structured/{experiment_name}_{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_f1", dirpath=ckpt_dir,
                        save_last=True, mode="max", save_top_k=1, save_weights_only=True)
    ]
    logger_dir = os.path.join(
        BASE_DIR, f"../../../data")
    os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="MGCA", save_dir=logger_dir, name=extension, offline=True)
    trainer = Trainer.from_argparse_args(
        args=args,
        callbacks=callbacks,
        logger=wandb_logger)

    model.training_steps = model.num_training_steps(trainer, datamodule)
    print(model.training_steps)
    trainer.fit(model, datamodule=datamodule)

    best_model_path = callbacks[1].best_model_path
    if best_model_path:
        print(f"Loading best model from {best_model_path} for testing.")
        trainer.test(datamodule=datamodule, ckpt_path=best_model_path)

    best_ckpt_path = os.path.join(ckpt_dir, "best_ckpts.yaml")

    training_record = {
        'experiment_info': {
            'name': experiment_name,
            'timestamp': extension,
            'epochs': trainer.current_epoch + 1,
            'steps': trainer.global_step,
            'hyperparameters': {
                'experiment_name': args.experiment_name,
                'img_encoder': args.img_encoder,
                'num_classes': args.num_classes,
                'learning_rate': args.learning_rate,
                'batch_size': args.batch_size,
                'lambda_cls': args.lambda_cls,
                'data_pct': args.data_pct,
            }
        },
        'test_results': getattr(model, 'final_test_results', {'status': 'No test completed'})
    }

    with open(best_ckpt_path, 'w', encoding='utf-8') as f:
        yaml.dump(training_record, f, default_flow_style=False, allow_unicode=True)

if __name__ == "__main__":
    cli_main()
