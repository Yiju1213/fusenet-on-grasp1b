import torch
from torch import Tensor

def split_rgbd_tensor(rgbd_tensor: Tensor) -> tuple[Tensor, Tensor]:
    """
    return:
    - (rgb_tensor, depth_tensor)
    """
    rgb_tensor = rgbd_tensor[:, :3, :, :]
    depth_tensor = rgbd_tensor[:, 3, :, :].unsqueeze(1)  # 添加通道维度

    return rgb_tensor, depth_tensor