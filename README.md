# fusenet-on-grasp1b
## 项目来源
- 机器学习课程大作业
## 项目内容
- 利用开源RGB-D语义分割网络[FuseNet](https://github.com/MehmetAygun/fusenet-pytorch)作为网络框架，利用开源机器人抓取场景数据集[Grasp-1Billion](https://graspnet.net/)作为数据集，尝试进行场景语义分割。
## 文件功能概述
- `data/dataloader.py` 继承Pytorch的Dataset类，实现从.pt格式的数据集中提取单个样本以及总样本数；
- `models/`
  - `accuracy.py` 实现语义分割三种准确度的计算(global_accu, mean_accu, iou_accu)
  - `network.py`/`utils.py` 定义FuseNet网络架构
- `visualization/` 实现各种可视化操作
- `run_time.ipynb` 基于上述功能文件具体实现数据预处理、网络训练、精度计算及结果可视化等流程
