import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# 生成随机的 RGB 颜色数组
def generate_random_colors(num_colors) -> np.ndarray:
    return np.random.randint(0, 256, size=(num_colors, 3), dtype=np.uint8).astype(np.float32) / 255 # [0, 1]

# 生成映射字典
def label2color_mapping(num_labels: int) -> dict:
    # 随机种子
    np.random.seed(42)
    # 字典推导 TODO 改成真实的名字
    label_names = {i: f'cls{i}' for i in range(num_labels)}
    # 生成可复现的随机颜色
    colors = generate_random_colors(num_labels)
    # 映射字典
    label_info = {}
    for label in range(num_labels):
        label_info[label] = {
            'color': colors[label],
            'name': label_names[label],
        }
    return label_info

# 可视化检查映射字典
def check_label2color_mapping(label_info: dict):
        # 设置图像大小
    num_rows = 9  # 因为 9 行 * 10 列 = 90 > 88
    fig, axes = plt.subplots(num_rows, 10, figsize=(10, 10))

    # 关闭坐标轴
    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')

    # 在每个 subplot 中显示类型名称和颜色
    for label, info in label_info.items():
        row = label // 10  # 行数
        col = label % 10   # 列数

        ax = axes[row, col]
        color = info['color']  # 将颜色值缩放到 [0, 1] 范围
        ax.imshow([[color]], aspect='auto')
        ax.set_title(info['name'])

    plt.show()