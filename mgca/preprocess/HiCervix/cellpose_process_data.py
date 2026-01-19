import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import uuid
import numpy as np
from cellpose import models, io


SOURCE_DIR = '.'
TARGET_DIR = '..'
IMAGES_TARGET_DIR = os.path.join(TARGET_DIR, 'cellpose_images')

# 数据集划分
DATASET_SETS = ['train', 'val', 'test']

# --- >>> 修改后的图片处理参数 <<< ---
TARGET_SIZE = 350
# 统一的初始缩放比例 (1.0 = 不进行缩放)
INITIAL_SHRINK_FACTOR = 1.0
# (新) “中心”区域的定义：只在图像中心的 75% 区域内搜索最热点
CENTRAL_SEARCH_RATIO = 0.75


def sigmoid(x):
    """计算 sigmoid 函数"""
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def process_image(img_path, class_name, cellpose_model, target_size=TARGET_SIZE, search_ratio=CENTRAL_SEARCH_RATIO):
    """
    1. 目标尺寸为 350x350。
    2. 严格不进行缩放 (1:1 像素保留)。
    3. (快速通道) 如果 W 和 H 都 <= 350，则只进行黑边填充，然后立即返回。
    4. (慢速通道) 否则 (对于大图或细长图)，执行“Pad-then-Biased-Hotspot-Crop”逻辑。
    """
    try:
        img_original = Image.open(img_path)
        if img_original.mode != 'RGB':
            img_original = img_original.convert('RGB')

        w_orig, h_orig = img_original.size
        original_name = os.path.splitext(os.path.basename(img_path))[0]

        if w_orig == 0 or h_orig == 0:
            return []

        if w_orig <= target_size and h_orig <= target_size:

            # 1. 创建 350x350 的画布
            background_pil = Image.new('RGB', (target_size, target_size), (0, 0, 0))

            # 2. 将原始图片粘贴到中心
            paste_x = (target_size - w_orig) // 2
            paste_y = (target_size - h_orig) // 2
            background_pil.paste(img_original, (paste_x, paste_y))

            # 3. 这张填充后的画布就是我们的最终图像
            final_image = background_pil
            new_name = f"{original_name}_padded_{target_size}.jpg"

        else:

            # 3.1: 确定“安全”画布的尺寸 (Pad 逻辑)
            w_safe = max(w_orig, target_size)
            h_safe = max(h_orig, target_size)

            # 3.2: 创建画布，并将原始图片粘贴到中心
            background_pil = Image.new('RGB', (w_safe, h_safe), (0, 0, 0))
            paste_x = (w_safe - w_orig) // 2
            paste_y = (h_safe - h_orig) // 2
            background_pil.paste(img_original, (paste_x, paste_y))

            # 3.3: 运行 Cellpose 获取热力图 (昂贵的步骤)
            img_np = np.array(background_pil)
            masks, flows, styles = cellpose_model.eval(img_np, channels=[0, 0])
            cellprob_prob = sigmoid(flows[2])  # 这是 h_safe x w_safe 的热力图

            # 3.4: 创建“中心偏置”搜索掩码
            prob_map_h, prob_map_w = cellprob_prob.shape
            center_y, center_x = prob_map_h // 2, prob_map_w // 2
            search_h = int(prob_map_h * search_ratio)
            search_w = int(prob_map_w * search_ratio)
            y_start = max(0, center_y - search_h // 2)
            y_end = min(prob_map_h, center_y + search_h // 2)
            x_start = max(0, center_x - search_w // 2)
            x_end = min(prob_map_w, center_x + search_w // 2)
            search_mask = np.zeros_like(cellprob_prob)
            search_mask[y_start:y_end, x_start:x_end] = 1
            biased_prob_map = cellprob_prob * search_mask

            # 3.5: 找到“中心区域”最热的那个像素 (Hotspot)
            hotspot_y, hotspot_x = np.unravel_index(np.argmax(biased_prob_map), biased_prob_map.shape)

            # 3.6: 以 Hotspot 为中心计算裁剪框
            top = hotspot_y - (target_size // 2)
            left = hotspot_x - (target_size // 2)

            # 3.7: “反向移动”以适配边界 (w_safe, h_safe)
            if top < 0:
                top = 0
            elif top + target_size > h_safe:
                top = h_safe - target_size
            if left < 0:
                left = 0
            elif left + target_size > w_safe:
                left = w_safe - target_size

            right = left + target_size
            bottom = top + target_size

            # 3.8: 裁剪
            final_image = background_pil.crop((left, top, right, bottom))
            new_name = f"{original_name}_biased_smart_crop_{target_size}.jpg"

        # --- 保存 (来自任一通道的结果) ---
        final_image.save(os.path.join(IMAGES_TARGET_DIR, new_name))

        return [{'image_name': new_name, 'class_name': class_name}]

    except Exception as e:
        print(f"处理 {img_path} 时发生严重错误: {e}")
        return []


def main():
    """
    主函数，执行所有处理步骤
    """
    # --- 2. 创建目标文件夹 ---
    print(f"创建目标文件夹: {IMAGES_TARGET_DIR}")
    os.makedirs(IMAGES_TARGET_DIR, exist_ok=True)

    # --- 3. 初始化 Cellpose 模型 ---
    print("正在初始化 Cellpose 模型 (这可能需要一些时间)...")
    cellpose_model = models.CellposeModel(gpu=True)
    print("Cellpose 模型已加载。")

    all_csv_data = {set_name: [] for set_name in DATASET_SETS}

    # --- 4. 遍历每个数据集 (train, val, test) ---
    for set_name in DATASET_SETS:
        print(f"\n--- 开始处理 '{set_name}' 数据集 ---")

        csv_path = os.path.join(SOURCE_DIR, f"{set_name}.csv")
        img_folder_path = os.path.join(SOURCE_DIR, set_name)

        if not os.path.exists(csv_path) or not os.path.exists(img_folder_path):
            print(f"警告: {csv_path} 或 {img_folder_path} 未找到，跳过该数据集。")
            continue

        df = pd.read_csv(csv_path)

        # 使用tqdm显示进度条
        progress_bar = tqdm(df.iterrows(), total=df.shape[0], desc=f"处理 {set_name} 图片")
        for _, row in progress_bar:
            img_name = row['image_name']
            class_name = row['class_name']
            img_path = os.path.join(img_folder_path, img_name)

            progress_bar.set_postfix_str(img_name, refresh=True)

            if os.path.exists(img_path):
                original_name = os.path.splitext(img_name)[0]

                # 1. 检查“快速通道”的预期输出文件
                expected_name_fast = f"{original_name}_padded_{TARGET_SIZE}.jpg"
                expected_path_fast = os.path.join(IMAGES_TARGET_DIR, expected_name_fast)

                # 2. 检查“慢速通道”的预期输出文件
                expected_name_slow = f"{original_name}_biased_smart_crop_{TARGET_SIZE}.jpg"
                expected_path_slow = os.path.join(IMAGES_TARGET_DIR, expected_name_slow)

                found_processed_file = None
                if os.path.exists(expected_path_fast):
                    found_processed_file = expected_name_fast
                elif os.path.exists(expected_path_slow):
                    found_processed_file = expected_name_slow

                if found_processed_file:
                    all_csv_data[set_name].append({'image_name': found_processed_file, 'class_name': class_name})
                    continue  # <-- 跳到下一个循环


                # 将模型实例传递给处理函数
                new_records = process_image(img_path, class_name, cellpose_model,
                                            target_size=TARGET_SIZE,
                                            search_ratio=CENTRAL_SEARCH_RATIO)
                if new_records:
                    all_csv_data[set_name].extend(new_records)
            else:
                progress_bar.write(f"警告: 图片未找到，已跳过: {img_path}")



    # --- 5. 生成所有新的CSV文件 ---
    print("\n--- 开始生成CSV文件 ---")
    all_data_list = []
    for set_name in DATASET_SETS:
        if all_csv_data[set_name]:
            new_df = pd.DataFrame(all_csv_data[set_name])
            output_path = os.path.join(TARGET_DIR, f"cellpose_{set_name}.csv")
            new_df.to_csv(output_path, index=False)
            print(f"已生成: {output_path} (包含 {len(new_df)} 条记录)")
            all_data_list.extend(all_csv_data[set_name])
        else:
            print(f"'{set_name}' 数据集没有生成任何数据，不创建CSV文件。")

    # 生成 all.csv
    if all_data_list:
        all_df = pd.DataFrame(all_data_list)
        output_path = os.path.join(TARGET_DIR, "cellpose_all.csv")
        all_df.to_csv(output_path, index=False)
        print(f"已生成: {output_path} (包含 {len(all_df)} 条记录)")

    print("\n处理完成！")


if __name__ == "__main__":
    main()