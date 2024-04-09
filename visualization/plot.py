import math
import torch
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np

def show_random_rgbd_imgs(rgbd_tensor: torch.Tensor, num_rows=5, num_cols=5):
    """
    随机展示一批RGBD图像。

    参数：
    - rgbd_tensor: 输入的RGBD张量，维度为 [数量, 4, 高度, 宽度]，包含RGB和深度信息
    - num_rows: 展示的行数
    - num_cols: 展示的列数
    """
    num_images = num_rows * num_cols
    indices = np.random.choice(len(rgbd_tensor), num_images, replace=False)

    fig, axes = plt.subplots(num_rows, 2 * num_cols, figsize=(20, 12))

    for i, ax in enumerate(axes.flat):
        index = math.floor(i / 2)

        rgbd_image = rgbd_tensor[indices[index]].numpy()

        # 提取RGB和深度图像
        rgb_image = rgbd_image[:3, :, :]
        depth_image = rgbd_image[3, :, :]

        if i % 2 == 0:
            # RGB图像展示
            ax.imshow(np.transpose(rgb_image, (1, 2, 0)))
            ax.axis('off')
            ax.set_title(f'RGB {index}')
        else:
            # 深度图像展示
            ax.imshow(depth_image, cmap='gray', vmin=depth_image.min(), vmax=depth_image.max())
            ax.axis('off')
            ax.set_title(f'Depth {index}')

    plt.show()


def visualize_predictions(rgb, mask, pred, label_info: dict, num_rows=30, num_cols=3):
    assert len(rgb) == len(mask) == len(pred), "Input lists must have the same length"

    # 设置图像大小
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))

    random_indices = np.random.choice(len(rgb), num_rows, replace=False)

    for i, idx in enumerate(random_indices):
        # 提取当前样本的 RGB、Mask 和 Pred
        cur_rgb: Tensor = rgb[idx]
        cur_mask: Tensor = torch.zeros_like(cur_rgb)
        cur_pred: Tensor = torch.zeros_like(cur_rgb)

        cur_rgb = cur_rgb.permute(1,2,0)
        cur_mask = cur_mask.permute(1,2,0)
        cur_pred = cur_pred.permute(1,2,0)

        for label, info in label_info.items():
            
            cur_mask[mask[idx] == label] = torch.tensor(info['color'])
            cur_pred[pred[idx] == label] = torch.tensor(info['color'])

        # 在subplot中显示图像
        axes[i, 0].imshow(cur_rgb)
        axes[i, 0].axis('off')
        axes[i, 1].imshow(cur_mask)
        axes[i, 1].axis('off')
        axes[i, 2].imshow(cur_pred)
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()