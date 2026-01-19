import cv2
import numpy as np
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


# 注意：下面的 DataTransforms 类将完全替换您文件中现有的同名类
# 其他类（如 Moco2Transform 等）可以保留，以备后用

class DataTransforms(object):
    def __init__(self, is_train: bool = True, crop_size: int = 224):

        # 验证和测试集只进行必要的尺寸调整和归一化
        if not is_train:
            self.transform = A.Compose([
                A.CenterCrop(height=crop_size, width=crop_size, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        # 训练集使用您提供的专业级数据增强流程
        else:
            self.transform = A.Compose([
                # 1. 首先进行随机裁剪，确定基础训练图像
                A.RandomCrop(height=crop_size, width=crop_size, p=1.0),

                # 2. 几何与结构变换
                A.RandomGridShuffle(grid=(4, 4), p=0.5),
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT,
                                   value=0, p=0.5),

                # 3. 颜色增强 (保守且有效)
                A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1, hue=0.05, p=0.3),

                # 4. 图像质量/噪声模拟
                A.Downscale(scale_max=0.5, scale_min=0.5, interpolation=cv2.INTER_AREA, p=0.15),
                A.Blur(blur_limit=7, p=0.2),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.25),

                # 5. 归一化并转换为Tensor
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    def __call__(self, image):
        # Albumentations 需要 NumPy array 作为输入
        image_np = np.array(image)
        # 应用增强
        transformed = self.transform(image=image_np)
        return transformed['image']
