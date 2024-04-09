import torch
import torch.nn as nn
import torch.utils.data as data
from torch import Tensor

from models.accuracy import calculate_accuracy
from data.utils import split_rgbd_tensor

# train
def train(model: nn.Module, iterator: data.DataLoader, 
          optimizer: torch.optim.Optimizer, criterion, device, tiny_set: bool)\
            -> tuple[float, float, float, float]:
    
    epoch_loss = 0
    epoch_glb_acc = 0
    epoch_mean_acc = 0
    epoch_iou_acc = 0
    
    model.train() # Train Mode: Enable Dropout, BatchNorm, ...
    
    for (rgbd, mask) in iterator:
        rgbd: Tensor
        mask: Tensor
        
        rgb, depth = split_rgbd_tensor(rgbd)
        rgb = rgb.to(device)
        depth = depth.to(device)
        mask = mask.to(device)
        
        # 清除其他网络训练时留下的梯度信息
        optimizer.zero_grad() 
        # 模型前向传播 此时计算平台构建动态计算图 用于后续反向传播时使用
        if tiny_set:
            y_pred: Tensor = model(rgb, depth, True)
        else:
            y_pred: Tensor = model(rgb, depth)
        # 通过定义的准则计算loss
        loss: Tensor = criterion(y_pred, mask, model)
        # # 计算准确率
        glb, mean, iou = calculate_accuracy(y_pred, mask)
        # 反向传播 调用计算平台动态构建的计算图 并传入loss数据 计算各参数梯度
        loss.backward()
        # 通过得到的梯度 以定义的优化算法更新 参数权重
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_glb_acc += glb.item()
        epoch_mean_acc += mean.item()
        epoch_iou_acc += iou.item()
    
    optimizer.zero_grad() 
    
        
    return epoch_loss / len(iterator), epoch_glb_acc / len(iterator), \
            epoch_mean_acc / len(iterator), epoch_iou_acc / len(iterator)

def evaluate(model: nn.Module, iterator: data.DataLoader, criterion, device)\
                -> tuple[float, float, float, float]:
    
    epoch_loss = 0
    epoch_glb_acc = 0
    epoch_mean_acc = 0
    epoch_iou_acc = 0

    # 1. turn on eval mode
    model.eval()

    with torch.no_grad():
        for (rgbd, mask) in iterator:
            rgbd: Tensor
            mask: Tensor
            
            # 2. transfer data to computation device(split channel)
            rgb, depth = split_rgbd_tensor(rgbd)
            rgb = rgb.to(device)
            depth = depth.to(device)
            mask = mask.to(device)
            # 3. forward propagation computation
            y_pred: Tensor = model(rgb, depth)
            # 4. compute loss
            loss: Tensor = criterion(y_pred, mask, model)
            # 5. compute acc
            glb, mean, iou = calculate_accuracy(y_pred, mask)
            # 6. add-on 
            epoch_loss += loss.item()
            epoch_glb_acc += glb.item()
            epoch_mean_acc += mean.item()
            epoch_iou_acc += iou.item()
        
    return epoch_loss / len(iterator), epoch_glb_acc / len(iterator), \
            epoch_mean_acc / len(iterator), epoch_iou_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def get_predictions(model: nn.Module, iterator: data.DataLoader, device) -> tuple[Tensor, Tensor, Tensor]:
    all_rgb = []
    all_mask = []
    all_pred = []

    model.eval()
    
    with torch.no_grad():
        for (batch_rgbd, batch_mask) in iterator:
            batch_rgbd: Tensor
            batch_mask: Tensor

            batch_rgb, batch_depth = split_rgbd_tensor(batch_rgbd)

            batch_rgb = batch_rgb.to(device)
            batch_depth = batch_depth.to(device)
            batch_mask = batch_mask.to(device)

            batch_pred = model(batch_rgb, batch_depth)
            batch_pred = torch.argmax(batch_pred, dim=1)

            all_rgb.append(batch_rgb)
            all_mask.append(batch_mask)
            all_pred.append(batch_pred)
    
    # cat list into a whole tensor
    all_rgb = torch.cat(all_rgb, dim=0) # 只有dim=0（行）参与运算（扩张行，列index不会变）
    all_mask = torch.cat(all_mask, dim=0)
    all_pred = torch.cat(all_pred, dim=0)

    return all_rgb, all_mask, all_pred



if __name__ == '__main__':
    pass
