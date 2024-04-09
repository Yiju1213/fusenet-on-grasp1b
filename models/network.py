from models.utils import *
import torch

class Fusenet(nn.Module):
    def __init__(self, num_labels, rgb_enc=True, depth_enc=True, rgb_dec=True, depth_dec=False):
        super(Fusenet, self).__init__()
        batchNorm_momentum = 0.1#TODO:make param

        self.need_initialization: list[nn.Sequential] = [] #modules that need initialization
        model_dic = VGG16_initializator()

        if rgb_enc :

            ##### RGB ENCODER ####
            self.CBR1_RGB_ENC = make_crb_layer_from_names(["conv1_1","conv1_2"], model_dic, 64)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True) # 224->112

            self.CBR2_RGB_ENC = make_crb_layer_from_names(["conv2_1","conv2_2"], model_dic, 128)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True) # 112->56

            self.CBR3_RGB_ENC = make_crb_layer_from_names(["conv3_1","conv3_2","conv3_3"], model_dic, 256)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True) # 56->28
            self.dropout3 = nn.Dropout(p=0.4)

            self.CBR4_RGB_ENC = make_crb_layer_from_names(["conv4_1","conv4_2","conv4_3"], model_dic, 512)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True) # 28->14
            self.dropout4 = nn.Dropout(p=0.4)

            self.CBR5_RGB_ENC = make_crb_layer_from_names(["conv5_1","conv5_2","conv5_3"], model_dic, 512)
            self.dropout5 = nn.Dropout(p=0.4)

            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True) # 14->7

        if depth_enc :
            feats_depth = list(torchvision.models.vgg16(pretrained=True).features.children())
            avg = torch.mean(feats_depth[0].weight.data, dim=1)
            avg = avg.unsqueeze(1)

            conv11d = nn.Conv2d(1, 64, kernel_size=3,padding=1)
            conv11d.weight.data = avg

            self.CBR1_DEPTH_ENC = make_crb_layer_from_names(["conv1_2"], model_dic, 64, conv11d)
            self.pool1_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

            self.CBR2_DEPTH_ENC = make_crb_layer_from_names(["conv2_1","conv2_2"], model_dic, 128)
            self.pool2_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

            self.CBR3_DEPTH_ENC = make_crb_layer_from_names(["conv3_1","conv3_2","conv3_3"], model_dic, 256)
            self.pool3_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
            self.dropout3_d = nn.Dropout(p=0.4)

            self.CBR4_DEPTH_ENC = make_crb_layer_from_names(["conv4_1","conv4_2","conv4_3"], model_dic, 512)
            self.pool4_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
            self.dropout4_d = nn.Dropout(p=0.4)

            self.CBR5_DEPTH_ENC = make_crb_layer_from_names(["conv5_1","conv5_2","conv5_3"], model_dic, 512)

        if  rgb_dec :
            ####  RGB DECODER  ####

            self.unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2) # 7->14
            self.CBR5_RGB_DEC = make_crb_layers_from_size([[512,512],[512,512],[512,512]])
            self.dropout5_dec = nn.Dropout(p=0.4)

            self.need_initialization.append(self.CBR5_RGB_DEC)

            self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2) # 14->28
            self.CBR4_RGB_DEC = make_crb_layers_from_size([[512,512],[512,512],[512,256]])
            self.dropout4_dec = nn.Dropout(p=0.4)

            self.need_initialization.append(self.CBR4_RGB_DEC)

            self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2) # 28->56
            self.CBR3_RGB_DEC = make_crb_layers_from_size([[256,256],[256,256],[256,128]])
            self.dropout3_dec = nn.Dropout(p=0.4)

            self.need_initialization.append(self.CBR3_RGB_DEC)

            self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2) # 56->112
            self.CBR2_RGB_DEC = make_crb_layers_from_size([[128,128],[128,64]]) # TODO 最后num_label是89 64需不需要改

            self.need_initialization.append(self.CBR2_RGB_DEC)

            self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2) # 112->224
            self.CBR1_RGB_DEC = nn.Sequential (
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96, momentum= batchNorm_momentum),
            nn.ReLU(),
            nn.Conv2d(96, num_labels, kernel_size=3, padding=1), # score
            )

            self.need_initialization.append(self.CBR1_RGB_DEC)

    def forward(self, rgb_inputs, depth_inputs, tiny: bool = False):

        ########  DEPTH ENCODER  ########
        # Stage 1
        #x = self.conv11d(depth_inputs)
        x_1 = self.CBR1_DEPTH_ENC(depth_inputs)
        x, id1_d = self.pool1_d(x_1)

        # Stage 2
        x_2 = self.CBR2_DEPTH_ENC(x)
        x, id2_d = self.pool2_d(x_2)

        # Stage 3
        x_3 = self.CBR3_DEPTH_ENC(x)
        x, id3_d = self.pool4_d(x_3)
        if not tiny:
            x = self.dropout3_d(x)

        # Stage 4
        x_4 = self.CBR4_DEPTH_ENC(x)
        x, id4_d = self.pool4_d(x_4)
        if not tiny:
            x = self.dropout4_d(x)

        # Stage 5
        x_5 = self.CBR5_DEPTH_ENC(x)

        ########  RGB ENCODER  ########

        # Stage 1
        y = self.CBR1_RGB_ENC(rgb_inputs)
        y = torch.add(y,x_1)
        y = torch.div(y,2)
        y, id1 = self.pool1(y)

        # Stage 2
        y = self.CBR2_RGB_ENC(y)
        y = torch.add(y,x_2)
        y = torch.div(y,2)
        y, id2 = self.pool2(y)

        # Stage 3
        y = self.CBR3_RGB_ENC(y)
        y = torch.add(y,x_3)
        y = torch.div(y,2)
        y, id3 = self.pool3(y)
        if not tiny:
            y = self.dropout3(y)

        # Stage 4
        y = self.CBR4_RGB_ENC(y)
        y = torch.add(y,x_4)
        y = torch.div(y,2)
        y, id4 = self.pool4(y)
        if not tiny:
            y = self.dropout4(y)

        # Stage 5
        y = self.CBR5_RGB_ENC(y)
        y = torch.add(y,x_5)
        y = torch.div(y,2)
        y_size = y.size()

        y, id5 = self.pool5(y)
        if not tiny:
            y = self.dropout5(y)

        ########  DECODER  ########

        # Stage 5 dec
        y = self.unpool5(y, id5,output_size=y_size)
        y = self.CBR5_RGB_DEC(y)
        if not tiny:
            y = self.dropout5_dec(y)

        # Stage 4 dec
        y = self.unpool4(y, id4)
        y = self.CBR4_RGB_DEC(y)
        if not tiny:
            y = self.dropout4_dec(y)

        # Stage 3 dec
        y = self.unpool3(y, id3)
        y = self.CBR3_RGB_DEC(y)
        if not tiny:
            y = self.dropout3_dec(y)

        # Stage 2 dec
        y = self.unpool2(y, id2)
        y = self.CBR2_RGB_DEC(y)

        # Stage 1 dec
        y = self.unpool1(y, id1)
        y = self.CBR1_RGB_DEC(y)

        return y
    

# 针对需要初始化的module进行参数初始化
def init_params(sequential_list: list[nn.Sequential]):
    for each_seq in sequential_list:
        # print(f"{each_seq}")
        for idx, m in enumerate(each_seq.children()):
            if isinstance(m, nn.Conv2d):
                # print(f"init Conv2d ({idx})")
                nn.init.kaiming_normal_(m.weight.data, nonlinearity = 'relu')
                nn.init.constant_(m.bias.data, 0)

import torch
import torch.nn as nn

class FusenetLoss(nn.Module):
    def __init__(self, alpha=1e-4):
        super(FusenetLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss() 
        self.alpha = alpha  # 调整正则化项的权重

    def forward(self, output, target, model: nn.Module):
        ce_loss = self.cross_entropy(output, target)
        
        # # 添加 L2 正则化项
        # l2_reg = torch.tensor(0., requires_grad=False, device='cuda')
        # l2_reg.to(device='cuda')
        # for param in model.parameters():
        #     param = param.to(device='cuda')
        #     l2_reg += torch.norm(param, p=2)
        # l2_reg = l2_reg.requires_grad_()

        # total_loss = ce_loss + self.alpha * l2_reg
        return ce_loss
