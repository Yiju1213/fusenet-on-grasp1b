import torch
from torch import Tensor
from sklearn.metrics import confusion_matrix

def calculate_accuracy(predict: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """
    计算图像分割任务中的三类准确率

    返回：
    (global_accu, mean_accu, iou_accu)
    """
    predict = torch.argmax(predict, dim=1)
    tar = target.to('cpu').detach().view(-1)
    pre = predict.to('cpu').detach().view(-1)
    cm = torch.tensor(confusion_matrix(tar, pre))
    glb = global_accuracy(cm)
    mean = mean_accuracy(cm)
    iou = iou_accuracy(cm)

    return (glb, mean, iou)

def global_accuracy(cm: Tensor) -> Tensor:
    """
    计算图像分割任务的全局准确率
    global = 1 / N * sum_{c} TP_c

    返回：
    - global_accuracy: 全局准确率
    """

    correct = cm.diag().sum()
    total = cm.sum()

    return correct / (total + 1e-10)

def mean_accuracy(cm: Tensor) -> Tensor:
    """
    计算图像分割任务的平均准确率。
    mean = 1 / K * sum_{c} TP_c / (TP_c + FP_c)

    返回：
    - global_accuracy: 全局准确率
    """
    # 计算每个类别的准确性
    tp = cm.diag()
    fp = cm.sum(dim=0) - tp
    class_accuracy = tp / (tp + fp + 1e-10)
    
    # 计算平均准确性
    mean_acc = class_accuracy.mean()
    
    return mean_acc

def iou_accuracy(cm: Tensor) -> Tensor:
    """
    计算 Intersection-over-Union (IoU)

    返回值：
    - iou: IoU 值，形状为 (num_classes,)
    """

    tp = cm.diag()
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp

    # 计算 IoU
    iou = tp / (tp + fp + fn + 1e-10)

    return iou.mean()