import cv2
import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- 1. 配置您的路径 ---
CSV_PATH = "/jeyzhang/leijiaxin/A_CellModel/Cell_Datasets/HiCervix/cellpose_all.csv"
SOURCE_IMAGE_DIR = "/jeyzhang/leijiaxin/A_CellModel/Cell_Datasets/HiCervix/cellpose_images"
TARGET_IMAGE_DIR = "/jeyzhang/leijiaxin/A_CellModel/Cell_Datasets/HiCervix/cellpose_images_256"

# --- 2. 图像处理配置 ---
TARGET_SIZE = 256
# 假设您的CSV中包含一个名为 'image_name' 的列
IMAGE_COLUMN_NAME = "image_name"
# 设置您想使用的CPU核心数 (None 表示使用所有核心)
NUM_WORKERS = None


# --- 3. 复制您“最好保留像素”的函数 ---
def resize_img(img, scale):
    """
    Args:
        img - image as numpy array (cv2)
        scale - desired output image-size as scale x scale
    Return:
        image resized to scale x scale with shortest dimension 0-padded
    """
    size = img.shape
    height, width = size[0], size[1]

    # Resizing
    if height > width:
        # 图像偏高
        new_height = scale
        new_width = int(width * (scale / height))
    else:
        # 图像偏宽或为方形
        new_width = scale
        new_height = int(height * (scale / width))

    # 使用 cv2.INTER_AREA (区域插值)，它被认为是缩小图像时保留细节最好的方法
    # resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA) #图像用
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST) #概率图用

    h_resized, w_resized = resized_img.shape[0], resized_img.shape[1]

    pad_h = scale - h_resized
    pad_w = scale - w_resized

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    # Padding
    if img.ndim == 3:
        # 对于彩色图像, 需为颜色通道指定(0, 0)的padding
        padding = [(top, bottom), (left, right), (0, 0)]
    else:
        # 对于灰度图像, 维持原样
        padding = [(top, bottom), (left, right)]

    # 使用 'constant' 和 0 填充
    padded_img = np.pad(
        resized_img, padding, "constant", constant_values=0
    )
    return padded_img


# --- 4. 单个图像的处理函数 (供多进程调用) ---
def process_single_image(image_filename):
    """
    加载、调整大小并保存单个图像。
    """
    image_filename = os.path.splitext(image_filename)[0] + ".png"
    source_path = os.path.join(SOURCE_IMAGE_DIR, image_filename)
    target_path = os.path.join(TARGET_IMAGE_DIR, image_filename)

    # 4.1. 如果目标文件已存在，则跳过 (支持断点续传)
    if os.path.exists(target_path):
        return f"Skipped: {image_filename} (already exists)"

    # 4.2. 读取源图像
    try:
        img = cv2.imread(source_path)
        if img is None:
            raise IOError(f"Failed to read image (img is None)")
    except Exception as e:
        return f"Error reading {source_path}: {e}"

    # 4.3. 执行缩放操作
    try:
        resized_padded_img = resize_img(img, TARGET_SIZE)
    except Exception as e:
        return f"Error resizing {image_filename}: {e}"

    # 4.4. 保存目标图像
    try:
        # cv2.imwrite 会自动处理 jpg, png 等格式
        cv2.imwrite(target_path, resized_padded_img)
        return f"Success: {image_filename}"
    except Exception as e:
        return f"Error writing {target_path}: {e}"


# --- 5. 主执行函数 ---
def main():
    print(f"--- 图像预处理脚本 ---")
    print(f"源 (Source)   : {SOURCE_IMAGE_DIR}")
    print(f"目标 (Target) : {TARGET_IMAGE_DIR}")
    print(f"CSV          : {CSV_PATH}")
    print(f"目标尺寸     : {TARGET_SIZE}x{TARGET_SIZE}\n")

    # 5.1. 创建目标文件夹
    if not os.path.exists(TARGET_IMAGE_DIR):
        print(f"创建目标文件夹: {TARGET_IMAGE_DIR}")
        os.makedirs(TARGET_IMAGE_DIR, exist_ok=True)
    else:
        print(f"目标文件夹已存在，将跳过已处理的图像。")

    # 5.2. 从 CSV 读取需要处理的文件列表
    try:
        df = pd.read_csv(CSV_PATH)
        if IMAGE_COLUMN_NAME not in df.columns:
            print(f"错误: CSV文件 {CSV_PATH} 中未找到列 '{IMAGE_COLUMN_NAME}'")
            sys.exit(1)
        # 获取唯一的图像文件名列表
        image_list = df[IMAGE_COLUMN_NAME].unique().tolist()
        print(f"从CSV中找到 {len(image_list)} 个唯一的图像文件。")
    except Exception as e:
        print(f"读取CSV文件 {CSV_PATH} 时出错: {e}")
        sys.exit(1)

    # 5.3. 使用 ProcessPoolExecutor 并行处理
    print(f"\n开始使用 {NUM_WORKERS or os.cpu_count()} 个CPU核心进行处理...")

    # 统计计数器
    success_count = 0
    skipped_count = 0
    error_count = 0

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # 提交所有任务
        futures = {executor.submit(process_single_image, filename): filename for filename in image_list}

        # 使用 tqdm 显示进度条
        progress = tqdm(as_completed(futures), total=len(image_list), desc="Processing Images")

        for future in progress:
            result = future.result()
            if result.startswith("Success"):
                success_count += 1
            elif result.startswith("Skipped"):
                skipped_count += 1
            else:
                error_count += 1
                # 实时打印错误信息
                print(f"\n[!] {result}", file=sys.stderr)

    # 5.4. 打印最终报告
    print("\n--- 处理完成 ---")
    print(f"  成功处理: {success_count}")
    print(f"  跳过 (已存在): {skipped_count}")
    print(f"  发生错误: {error_count}")
    print(f"总计: {success_count + skipped_count + error_count}")
    print("------------------")
    if error_count > 0:
        print("请检查上面列出的错误信息。")


if __name__ == "__main__":
    main()