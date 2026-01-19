import cv2
import numpy as np
import pydicom
from PIL import Image
from mgca.constants import *


def read_from_dicom(img_path, imsize=None, transform=None):
    dcm = pydicom.read_file(img_path)
    x = dcm.pixel_array

    x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))
    if dcm.PhotometricInterpretation == "MONOCHROME1":
        x = cv2.bitwise_not(x)

    # transform images
    if imsize is not None:
        x = resize_img(x, imsize)

    img = Image.fromarray(x).convert("RGB")

    if transform is not None:
        img = transform(img)

    return img


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
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

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

    padded_img = np.pad(
        resized_img, padding, "constant", constant_values=0
    )
    return padded_img


def get_imgs(img_path, scale, transform=None, multiscale=False):
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    x = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # tranform images
    x = resize_img(x, scale)
    img = Image.fromarray(x).convert("RGB")
    if transform is not None:
        img = transform(img)

    return img
