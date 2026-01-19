import pandas as pd
import os
from sklearn.model_selection import train_test_split

# ================= 配置区域 =================

INPUT_CSV_PATH = "all.csv"

# 2. 输出目录
OUTPUT_DIR = "split_data_6cls"

# 3. 目标类别
TARGET_CLASSES = ["ASC-H", "ASC-US", "HSIL", "LSIL", "NILM", "SCC"]

# 4. 随机种子 (保证可复现)
RANDOM_SEED = 42


# ===========================================

def process_csv():
    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    print(f"Loading data from: {INPUT_CSV_PATH}")
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Input CSV not found at {INPUT_CSV_PATH}")
        return

    print(f"Original total samples: {len(df)}")

    df_filtered = df[df['class_name'].isin(TARGET_CLASSES)].copy()

    print(f"Samples after removing NILM: {len(df_filtered)}")
    print("-" * 30)

    # --- 2. 数据集划分 (6:2:2 Stratified Splitting) ---

    # 目标比例：Train(60%), Val(20%), Test(20%)

    df_train_val, df_test = train_test_split(
        df_filtered,
        test_size=0.2,  # 20% 用于测试
        random_state=RANDOM_SEED,
        stratify=df_filtered['class_name']
    )

    df_train, df_val = train_test_split(
        df_train_val,
        test_size=0.25,  # 相对剩余数据的 25%，即总数据的 20%
        random_state=RANDOM_SEED,
        stratify=df_train_val['class_name']
    )

    # --- 3. 保存文件 ---
    train_save_path = os.path.join(OUTPUT_DIR, "train.csv")
    val_save_path = os.path.join(OUTPUT_DIR, "val.csv")
    test_save_path = os.path.join(OUTPUT_DIR, "test.csv")

    df_train.to_csv(train_save_path, index=False)
    df_val.to_csv(val_save_path, index=False)
    df_test.to_csv(test_save_path, index=False)

    # --- 4. 打印统计验证 ---
    print("Final Split Statistics (Target 6:2:2):")
    print(f"Train set: {len(df_train)} samples saved to {train_save_path}")
    print(f"Val set:   {len(df_val)} samples saved to {val_save_path}")
    print(f"Test set:  {len(df_test)} samples saved to {test_save_path}")

    print("-" * 30)
    print("Class Distribution in Test Set (Check SCC > 30):")
    print(df_test['class_name'].value_counts())


if __name__ == "__main__":
    process_csv()