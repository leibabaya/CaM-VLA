import os
import pickle
import re
import json
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from nltk.tokenize import RegexpTokenizer
from mgca.constants import *
from mgca.datasets.utils import get_imgs
from tqdm import tqdm
from transformers import BertTokenizer
from flatten_json import flatten
from sklearn.model_selection import train_test_split
import shutil

class MultimodalPretrainingDataset(data.Dataset):
    def __init__(self, split="train", transform=None, data_pct=1.0,
                 imsize=256, max_words=256, sent_num=3):
        super().__init__()
        self.transform = transform
        self.imsize = imsize
        self.max_words = max_words

        self.attr_max_words = 25
        self.attribute_order = [
            "N/C Ratio",
            "Chromatin",
            "Chromatin Distribution",
            "Nuclear Membrane",
            "Mitotic Figures",
            "Cell Arrangement",
            "Cell Structure",
            "Nuclear Staining",
            "Nuclear Size/Area",
            "Background",
            "Special Features",
            "Cytoplasm Amount",
            "Cytoplasm Texture",
            "Cytoplasm Staining",
            "Cytoplasm Contents",
            "Basic Cell Shape",
            "Special Cell Form",
            "Relative Cell Size",
            "Cell Size Consistency",
            "Nucleoli",
            "Multinucleation/Binucleation",
            "Cell Borders"
        ]
        self.num_attributes = len(self.attribute_order)
        print(
            f"Attribute extraction mode: Per-Attribute. Total attributes: {self.num_attributes}, Max len per attr: {self.max_words}")

        self.image_dir = "/XXX/XXX/XXX/XXX/HiCervix/cellpose_images_256"
        train_csv_path = "/XXX/XXX/XXX/XXX/HiCervix/cellpose_train_21class.csv"
        val_csv_path = "/XXX/XXX/XXX/XXX/HiCervix/cellpose_val_21class.csv"
        test_csv_path = "/XXX/XXX/XXX/XXX/HiCervix/cellpose_test_21class.csv"

        json_path = "/XXX/XXX/XXX/XXX/structured_morphological_description.json"
        print(json_path)
        attr_json_path = "/XXX/XXX/XXX/XXX/structured_morphological_description.json"
        print(attr_json_path)

        if not os.path.exists(self.image_dir):
            raise RuntimeError(f"Image directory not found at calculated path: {self.image_dir}")
        if not os.path.exists(train_csv_path):
            raise RuntimeError(f"Train CSV file not found at calculated path: {train_csv_path}")
        if not os.path.exists(val_csv_path):
            raise RuntimeError(f"Val CSV file not found at calculated path: {val_csv_path}")
        if not os.path.exists(test_csv_path):
            raise RuntimeError(f"Test CSV file not found at calculated path: {test_csv_path}")
        if not os.path.exists(json_path):
            raise RuntimeError(f"JSON description file not found at: {json_path}")
        if not os.path.exists(attr_json_path):
            raise RuntimeError(f"JSON description file not found at: {attr_json_path}")

        # --- 3. LOAD AND PREPARE DATA (EFFICIENT METHOD) ---
        train_df_full = pd.read_csv(train_csv_path)
        val_df_full = pd.read_csv(val_csv_path)
        test_df_full = pd.read_csv(test_csv_path)

        with open(json_path, 'r', encoding='utf-8') as f:
            nested_descriptions = json.load(f)
        class_to_description_str = {}
        for class_name, nested_dict in nested_descriptions.items():
            flat_dict = flatten(nested_dict, ' ')  # Using space as a separator for readability
            description_parts = [f"{key}: {value}." for key, value in flat_dict.items()]
            class_to_description_str[class_name] = " ".join(description_parts)
        for class_name, description in class_to_description_str.items():
            print(f"KEY: {class_name}\n")
            print(f"VALUE: {description}\n")

        with open(attr_json_path, 'r', encoding='utf-8') as f:
            descriptions_by_class = json.load(f)
        self.class_to_attributes = {}
        for class_name, attributes in descriptions_by_class.items():
            processed_attrs = {}
            for attr_key, attr_value in attributes.items():
                # 处理可能存在的嵌套字典或直接取值
                if isinstance(attr_value, dict):
                    parts = [f"{k} {v}" for k, v in attr_value.items()]  # 简化嵌套
                    value_text = ", ".join(parts)
                else:
                    value_text = str(attr_value)
                processed_attrs[attr_key] = f"{attr_key}: {value_text}"
            self.class_to_attributes[class_name] = processed_attrs

        # Step 3.3: Create a mapping from class name to a numerical index (label)
        all_classes_df = pd.concat([train_df_full, val_df_full, test_df_full])
        self.classes = sorted(all_classes_df['class_name'].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}
        print(f"Found {len(self.classes)} classes globally. Mapping: {self.class_to_idx}")

        def process_dataframe(df, name):
            df['label'] = df['class_name'].map(self.class_to_idx)
            original_rows = len(df)
            df.dropna(subset=['label'], inplace=True)
            if len(df) < original_rows:
                print(f"Warning: Dropped {original_rows - len(df)} rows from '{name}' due to missing labels.")
            df['label'] = df['label'].astype(int)
            return df

        train_df = process_dataframe(train_df_full, "Train")
        val_df = process_dataframe(val_df_full, "Validation")
        test_df = process_dataframe(test_df_full, "Test")

        if split == 'train':
            self.df = train_df

            print("--- Balancing Training Set via Under-sampling ---")
            print(f"Setting max samples per class to: {1000}")
            print("Original training set class distribution:")
            print(self.df['label'].value_counts().sort_index())

            # 使用 groupby 和 apply 实现高效的欠采样
            # 对于每个组，如果样本数超过上限，就随机采样；否则，取全部样本。
            self.df = self.df.groupby('label', group_keys=False).apply(
                lambda x: x.sample(n=min(len(x), 1000), random_state=42)
            )

            print("\nBalanced training set class distribution:")
            print(self.df['label'].value_counts().sort_index())
            print("-------------------------------------------------")


            # Correctly apply data percentage reduction only to the training set
            if data_pct != 1.0:
                print(f"Using {data_pct * 100:.2f}% of the training data.")
                self.df, _ = train_test_split(
                    self.df,
                    train_size=data_pct,  # 或者用 frac=data_pct, 但 train_size 更清晰
                    random_state=42,
                    stratify=self.df['label']  # 关键：告诉函数根据标签进行分层
                )
        elif 'val' in split:
            self.df = val_df
        else:
            self.df = test_df

        self.df = self.df.reset_index(drop=True)

        self.tokenizer = BertTokenizer.from_pretrained(
            "/XXX/XXX/XXX/XXX/pretrained_models/PathologyBERT")

        print("Pre-tokenizing text for ALL classes (Creating Global Tensors)...")
        # Coarse (粗粒度)
        coarse_input_ids, coarse_masks, coarse_types, coarse_lens = [], [], [], []
        # Fine (细粒度)
        fine_input_ids, fine_masks, fine_types, fine_lens = [], [], [], []
        for class_name in self.classes:
            # === 5.1 处理 Coarse (V1) - 单句 [1, 256] ===
            sent = class_to_description_str.get(class_name, "")
            c_tokens = self.tokenizer(sent, return_tensors="pt", truncation=True,
                                      padding="max_length", max_length=self.max_words)

            # squeeze(0) 去掉 batch 维度，稍后 stack
            coarse_input_ids.append(c_tokens['input_ids'].squeeze(0))
            coarse_masks.append(c_tokens['attention_mask'].squeeze(0))
            coarse_types.append(c_tokens['token_type_ids'].squeeze(0))
            coarse_lens.append(torch.tensor(c_tokens['attention_mask'].sum().item()))

            # === 5.2 处理 Fine (V2) - 属性组 [22, 25] ===
            attr_ids_list, attr_masks_list, attr_types_list = [], [], []
            attr_dict = self.class_to_attributes.get(class_name, {})

            for attr_key in self.attribute_order:
                text = attr_dict.get(attr_key, f"{attr_key}: Not applicable")
                f_tokens = self.tokenizer(text, padding='max_length', truncation=True,
                                          max_length=self.attr_max_words, return_tensors='pt')
                attr_ids_list.append(f_tokens['input_ids'])
                attr_masks_list.append(f_tokens['attention_mask'])
                attr_types_list.append(f_tokens['token_type_ids'])

            # 堆叠当前类别的 22 个属性 -> (22, 25)
            fine_input_ids.append(torch.cat(attr_ids_list, dim=0))
            fine_masks.append(torch.cat(attr_masks_list, dim=0))
            fine_types.append(torch.cat(attr_types_list, dim=0))
            fine_lens.append(torch.cat(attr_masks_list, dim=0).sum(dim=1))

            # === 6. 生成全局 Tensor Cache ===

        # V1 Coarse: Shape [Num_Classes, 256]
        self.tokenized_descriptions = {
            "input_ids": torch.stack(coarse_input_ids),
            "attention_mask": torch.stack(coarse_masks),
            "token_type_ids": torch.stack(coarse_types),
            "cap_lens": torch.stack(coarse_lens)
        }

        # V2 Fine: Shape [Num_Classes, 22, 25]
        self.attr_tokenized_descriptions = {
            "input_ids": torch.stack(fine_input_ids),
            "attention_mask": torch.stack(fine_masks),
            "token_type_ids": torch.stack(fine_types),
            "cap_lens": torch.stack(fine_lens)
        }

        print(f"Dataset Initialization Complete.")
        print(f"  -> Coarse Features: {self.tokenized_descriptions['input_ids'].shape} (For Classification)")
        print(f"  -> Fine Features:   {self.attr_tokenized_descriptions['input_ids'].shape} (For Alignment)")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_filename = row['image_name']
        image_path = os.path.join(self.image_dir, image_filename)
        label = row['label']
        imgs = get_imgs(image_path, self.imsize, self.transform, multiscale=False)

        return imgs, image_path, label, self.tokenized_descriptions, self.attr_tokenized_descriptions

def multimodal_collate_fn(batch):
    """sort sequence"""
    imgs, labels = [], []
    path = []
    tokenized_descriptions = None
    attr_tokenized_descriptions = None

    for b in batch:
        img, p, label, tok_desc, attr_tok_desc = b
        imgs.append(img)
        path.append(p)
        labels.append(label)
        if tokenized_descriptions is None:
            tokenized_descriptions = tok_desc
        if attr_tokenized_descriptions is None:
            attr_tokenized_descriptions = attr_tok_desc

    # stack
    imgs = torch.stack(imgs)
    labels = torch.tensor(labels, dtype=torch.long)
    path = np.array(path)

    return_dict = {
        "imgs": imgs,
        "path": path,
        "labels": labels,
        "tokenized_descriptions": tokenized_descriptions,
        "attr_tokenized_descriptions": attr_tokenized_descriptions
    }
    return return_dict

if __name__ == "__main__":
    from mgca.datasets.transforms import DataTransforms
    transform = DataTransforms(is_train=True)
    dataset = MultimodalPretrainingDataset(split="train", transform=transform)

