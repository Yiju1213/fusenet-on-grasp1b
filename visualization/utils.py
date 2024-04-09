import torch
import numpy as np
from PIL import Image
from torch import Tensor

# 将预测的label映射成RGBA
def visualize_predictions(rgb_image, predictions: Tensor):
    # 将预测值转化为标签（假设predictions是模型输出的张量）
    _, predicted_labels = torch.max(predictions, dim=1)
    predicted_labels = predicted_labels.squeeze().cpu().numpy()

    # 将标签映射为RGBA颜色
    colormap = create_colormap()  # 创建颜色映射表，函数定义见下面
    colored_mask = colormap[predicted_labels]

    # 将RGB图像转换为numpy数组
    rgb_array = np.array(rgb_image)

    # 叠加颜色掩码在RGB图像上
    overlaid_image = rgb_array.copy()
    overlaid_image[predicted_labels > 0] = colored_mask[predicted_labels > 0]

    # 显示或保存可视化结果
    Image.fromarray(overlaid_image).show()

def create_colormap():
    # 创建颜色映射表，这里简单地使用随机颜色，实际应根据类别数设计一种颜色映射方案
    num_classes = 89  # 88类别 + 1背景
    colormap = np.random.randint(0, 256, (num_classes, 4), dtype=np.uint8)
    colormap[:, 3] = 128  # 设置透明度
    return colormap

# 使用示例
# rgb_image 是RGB图像，predictions 是模型输出的预测结果
# visualize_predictions(rgb_image, predictions)
