import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import Tensor
import os

# 自定义数据集类
class FusenetDataset(Dataset):
    def __init__(self, rgbd_path: str, mask_path: str):
        # 在初始化中加载数据或设置文件路径等
        self.rgbd: Tensor = torch.load(rgbd_path)
        self.mask: Tensor = torch.load(mask_path)
        self.mask = self.mask.squeeze(1) # 变成三维
        self.mask = self.mask * 255 
        self.mask = self.mask.long()

    def __len__(self):
        # 返回数据集的大小
        return len(self.rgbd)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        # 根据索引加载并返回单个样本
        data = self.rgbd[idx]
        label = self.mask[idx]
        return (data, label)

    def get_rgbd_set(self) -> Tensor:
        pass
        return self.rgbd

    def get_mask_set(self) -> Tensor:
        pass
        return self.mask
    
    # def enable_tiny_set(self, tiny_set_num: int):
    #     self.backup_rgbd = self.rgbd
    #     self.backup_mask = self.mask
        
    
if __name__ == '__main__':

    rgbd_path = ('fusenet/yiju/dataset/train/rgbd.pt')
    mask_path = ('fusenet/yiju/dataset/train/mask.pt')
    
    custom_dataset = FusenetDataset(rgbd_path, mask_path)
    rgbd = custom_dataset.get_rgbd_set()
    print(custom_dataset.__len__())

    batch_size = 30
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

    i = 0
    for x, y in data_loader:
        print(i)
        print(x.shape)
        print(y.shape)
        i += 1
