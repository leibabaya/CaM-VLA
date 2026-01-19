import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support,
    multilabel_confusion_matrix, roc_auc_score, accuracy_score
)
from tabulate import tabulate
from torchmetrics import Metric


class DataCollector(Metric):
    """自定义 TorchMetrics，用于收集和同步数据"""

    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds, targets):
        self.preds.append(preds.detach())
        self.targets.append(targets.detach())

    def compute(self):

        def to_tensor(data):
            return data.cpu() if isinstance(data, torch.Tensor) else torch.cat(data, dim=0).cpu()

        return to_tensor(self.preds), to_tensor(self.targets)


class SimpleMetricsTable:
    def __init__(self, targets, probs, class_names):
        """
        简单的评估指标表格生成器

        Args:
            targets: self._test_targets (list of tensors)
            probs: self._test_probs (list of tensors)
            class_names: self.class_names (list)
        """
        self.label = torch.cat(targets, dim=0).numpy()
        self.pred = torch.cat(probs, dim=0).numpy()
        self.pred_labels = np.argmax(self.pred, axis=1)
        self.class_names = class_names
        self.num_classes = self.pred.shape[1]

    def get_table_and_auc(self):
        """
        返回格式化的表格字符串和AUC值

        Returns:
            tuple: (table_string, auc_value)
        """
        # 计算混淆矩阵
        cm = confusion_matrix(self.label, self.pred_labels, labels=np.arange(self.num_classes))
        print("混淆矩阵:\n", cm)

        # 计算各类指标
        print("\n每类指标:")
        prec, rec, f1, support = precision_recall_fscore_support(
            self.label, self.pred_labels, labels=np.arange(self.num_classes), average=None
        )

        # 计算每类准确率
        mcm = multilabel_confusion_matrix(self.label, self.pred_labels, labels=np.arange(self.num_classes))
        per_class_acc = []
        for i in range(self.num_classes):
            TN, FP, FN, TP = mcm[i].ravel()
            acc_class = (TP + TN) / cm.sum()  # 每类的真正accuracy
            per_class_acc.append(acc_class)
            print(f"\n类 {i}:")
            print(f"  acc={acc_class:.4f}, prec={prec[i]:.4f}, "
                  f"recall={rec[i]:.4f}, f1={f1[i]:.4f}, support={support[i]}")
            print("  混淆矩阵:")
            print(np.array([[TP, FP],
                            [FN, TN]]))

        # 计算整体指标
        y_true_onehot = np.eye(self.num_classes)[self.label.astype(int)]
        auc_macro = roc_auc_score(y_true_onehot, self.pred, multi_class="ovr", average="macro")
        overall_acc = accuracy_score(self.label, self.pred_labels)
        print("\n宏平均 AUC:", auc_macro)

        # 构建表格数据
        headers = ["Class", "Accuracy", "Precision", "Recall", "F1-Score", "Support"]
        table_data = []

        # 每个类别的数据
        for i in range(self.num_classes):
            table_data.append([
                self.class_names[i] if i < len(self.class_names) else f"Class {i}",
                f"{per_class_acc[i]:.4f}",
                f"{prec[i]:.4f}",
                f"{rec[i]:.4f}",
                f"{f1[i]:.4f}",
                f"{support[i]}"
            ])

        # 分隔线
        table_data.append(["-" * 8, "-" * 8, "-" * 9, "-" * 6, "-" * 8, "-" * 7])
        table_data.append(["Class", "Accuracy", "Precision", "Recall", "F1-Score", "Support"])

        # 宏平均
        table_data.append([
            "Macro Avg",
            f"{np.mean(per_class_acc):.4f}",
            f"{np.mean(prec):.4f}",
            f"{np.mean(rec):.4f}",
            f"{np.mean(f1):.4f}",
            f"{np.sum(support)}"
        ])

        # 微平均
        table_data.append([
            "Micro Avg",
            f"{overall_acc:.4f}",
            "-", "-", "-",
            f"{np.sum(support)}"
        ])

        # 生成表格字符串
        table_str = "=" * 80 + "\n"
        table_str += "分类评估指标汇总表\n"
        table_str += "=" * 80 + "\n"
        table_str += tabulate(table_data, headers=headers, tablefmt="grid") + "\n"
        table_str += f"宏平均 AUC: {auc_macro:.4f}\n"
        table_str += "=" * 80

        return table_str, auc_macro
