#===============================================================
#  A high-precision edge detection model
#  guided by flow field was proposed
#  by Yuchen Han, Bing Li et.al.
#  Some of the codes have not been disclosed yet,
#  but will be made public after the paper is accepted
#===============================================================

import math
import numpy as np
from .Snack_Conv import DSConv_pro
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ops import Conv2d
from .config import config_model, config_model_converted

class FBM(nn.Module):    #Feature broadcast module
    def __init__(self, inplane, outplane):
        super(FBM, self).__init__()
        self.get_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.get_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.ffg = nn.Conv2d(outplane * 2, 2, kernel_size=3, padding=1, bias=False)
    def forward(self, x):
        l_feature, h_feature = x
        h_feature_orign = h_feature
        h, w = l_feature.size()[2:]
        size = (h, w)
        l_feature = self.get_l(l_feature)
        h_feature = self.get_h(h_feature)
        h_feature = F.interpolate(h_feature, size=size, mode="bilinear", align_corners=False)
        flow_field = self.ffg(torch.cat([h_feature, l_feature], 1))
        h_feature = self.warp(h_feature_orign, flow_field, size=size)
        return h_feature

    @staticmethod
    def warp(inputs, flow_field, size):   #inputs是低分辨率
        out_h, out_w = size  # 对应高分辨率的low-level feature的特征图尺寸
        n, c, h, w = inputs.size()  # 对应低分辨率的high-level feature的4个输入维度
        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(inputs).to(inputs.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(inputs).to(inputs.device)
        grid = grid + flow_field.permute(0, 2, 3, 1) / norm
        output = F.grid_sample(inputs, grid)
        return output

class ABSAM(nn.Module):
    def __init__(self, channels):
        super(ABSAM, self).__init__()
        mid_channels = 6
        self.relu1 = nn.ReLU()
        self.rfc = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, padding=0)
        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1, padding=0)
        self.sac_1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(1, 3), padding=(0, 1))
        self.sac_2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(3, 1), padding=(1, 0))
        self.rfp = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.fg = nn.Conv2d(mid_channels, mid_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.conv1.bias, 0)
    def forward(self, x):
        y = self.relu1(x)
        y = self.conv1(y)
        peripheral = self.rfp(y)
        center = self.rfc(y)
        gap = nn.functional.adaptive_avg_pool2d(y, (1, 1))
        sac_out = self.sac_2(self.sac_1(y))
        ff = self.fg(gap)
        h_weights = peripheral * ff
        v_weights = sac_out * (1 - ff)
        outputs = h_weights + v_weights-center
        y = self.conv2(outputs)
        y = self.sigmoid(y)
        return x * y


class CDCM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CDCM, self).__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=5, padding=5, bias=False)
        self.conv2_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=7, padding=7, bias=False)
        self.conv2_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=9, padding=9, bias=False)
        self.conv2_4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=11, padding=11, bias=False)
        nn.init.constant_(self.conv1.bias, 0)
    def forward(self, x):
        x = self.relu1(x)
        x = self.conv1(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x4 = self.conv2_4(x)
        return x1 + x2 + x3 + x4

class MapReduce(nn.Module):
    def __init__(self, channels):
        super(MapReduce, self).__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, padding=0)
        nn.init.constant_(self.conv.bias, 0)
    def forward(self, x):
        return self.conv(x)

class PDCBlock(nn.Module):
    def __init__(self, pdc, inplane, ouplane, stride=1):
        super(PDCBlock, self).__init__()
        self.stride = stride
        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1,padding=0)  ##注意：nn.Conv2d是pytorch官方卷积。Conv2d是ops.py中的卷积，用于像素差卷积运算
        self.conv1 = Conv2d(pdc, inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)
        if self.stride > 1:
            x = self.shortcut(x)
        y = y + x
        return y

class PDCBlock_converted(nn.Module):

    def __init__(self, pdc, inplane, ouplane, stride=1):
        super(PDCBlock_converted, self).__init__()
        self.stride = stride

        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)
        if pdc == 'rd':
            self.conv1 = nn.Conv2d(inplane, inplane, kernel_size=5, padding=2, groups=inplane, bias=False)
        else:
            self.conv1 = nn.Conv2d(inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)
        if self.stride > 1:
            x = self.shortcut(x)
        y = y + x
        return y

class FFED(nn.Module):
    def __init__(self, inplane, pdcs, dil=None, sa=False, convert=False):
        super(FFED, self).__init__()
        self.sa = sa
        if dil is not None:
            assert isinstance(dil, int), 'dil should be an int'
        self.dil = dil
        self.fuseplanes = []
        self.inplane = inplane
        if convert:
            if pdcs[0] == 'rd':
                init_kernel_size = 5
                init_padding = 2
            else:
                init_kernel_size = 3
                init_padding = 1
            self.init_block = nn.Conv2d(3, self.inplane, kernel_size=init_kernel_size, padding=init_padding, bias=False)
            block_class = PDCBlock_converted  # 使用PCB列表
            block_res2_class = PDCBlock_converted
        else:
            self.init_block = Conv2d(pdcs[0], 3, self.inplane, kernel_size=3, padding=1)
            block_class = PDCBlock
            block_res2_class = PDCBlock_converted
        self.block1_1 = block_class(pdcs[1], self.inplane, self.inplane)
        self.block1_2 = block_class(pdcs[2], self.inplane, self.inplane)
        self.block1_3 = block_class(pdcs[3], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # C
        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block2_1 = block_class(pdcs[4], inplane, self.inplane, stride=2)
        self.block2_2 = block_class(pdcs[5], self.inplane, self.inplane)
        self.block2_3 = block_class(pdcs[6], self.inplane, self.inplane)
        self.block2_4 = block_class(pdcs[7], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # 2C
        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block3_1 = block_class(pdcs[8], inplane, self.inplane, stride=2)
        self.block3_2 = block_class(pdcs[9], self.inplane, self.inplane)
        self.block3_3 = block_class(pdcs[10], self.inplane, self.inplane)
        self.block3_4 = block_class(pdcs[11], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # 4C
        self.block4_1 = block_class(pdcs[12], self.inplane, self.inplane, stride=2)
        self.block4_2 = block_class(pdcs[13], self.inplane, self.inplane)
        self.block4_3 = block_class(pdcs[14], self.inplane, self.inplane)
        self.block4_4 = block_class(pdcs[15], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # 4C
        self.x1_to_240 = nn.Conv2d(60, 240, kernel_size=1, padding=0, bias=False)
        self.x2_to_240 = nn.Conv2d(120, 240, kernel_size=1, padding=0, bias=False)
        self.fbm_x1_x4 = FBM(inplane=240,outplane=240)
        self.fbm_x1_x3 = FBM(inplane=240,outplane=240)
        self.fbm_x1_x2 = FBM(inplane=240,outplane=240)
        self.fbm_x2_x3 = FBM(inplane=240, outplane=240)
        self.fbm_x2_x4 = FBM(inplane=240,outplane=240)
        self.fbm_x3_x4 = FBM(inplane=240,outplane=240)
        self.fuseplanes = [240,240,240,240,240,240]
        self.conv_reduces = nn.ModuleList()
        if self.sa and self.dil is not None:
            self.attentions = nn.ModuleList()
            self.dilations = nn.ModuleList()
            for i in range(6):
                self.dilations.append(CDCM(self.fuseplanes[i], self.dil))  # fuseplanes就是每部分block的输出，一共四个
                self.attentions.append(ABSAM(self.dil))
                self.conv_reduces.append(MapReduce(self.dil))  # MapReduce就是1*1卷积
        elif self.sa:
            self.attentions = nn.ModuleList()
            for i in range(4):
                self.attentions.append(ABSAM(self.fuseplanes[i]))
                self.conv_reduces.append(MapReduce(self.fuseplanes[i]))
        elif self.dil is not None:
            self.dilations = nn.ModuleList()
            for i in range(4):
                self.dilations.append(CDCM(self.fuseplanes[i], self.dil))
                self.conv_reduces.append(MapReduce(self.dil))
        else:
            for i in range(4):
                self.conv_reduces.append(MapReduce(self.fuseplanes[i]))
        self.classifier = nn.Conv2d(6, 1, kernel_size=1)  # has bias
        nn.init.constant_(self.classifier.weight, 0.25)
        nn.init.constant_(self.classifier.bias, 0)
        print('initialization done')

    def get_weights(self):
        conv_weights = []
        bn_weights = []
        relu_weights = []
        for pname, p in self.named_parameters():
            if 'bn' in pname:
                bn_weights.append(p)
            elif 'relu' in pname:
                relu_weights.append(p)
            else:
                conv_weights.append(p)

        return conv_weights, bn_weights, relu_weights

    def forward(self, x):
        H, W = x.size()[2:]

        x = self.init_block(x)

        x1 = self.block1_1(x)
        x1 = self.block1_2(x1)
        x1 = self.block1_3(x1)

        x2 = self.block2_1(x1)
        x2 = self.block2_2(x2)
        x2 = self.block2_3(x2)
        x2 = self.block2_4(x2)

        x3 = self.block3_1(x2)
        x3 = self.block3_2(x3)
        x3 = self.block3_3(x3)
        x3 = self.block3_4(x3)

        x4 = self.block4_1(x3)
        x4 = self.block4_2(x4)
        x4 = self.block4_3(x4)
        x4 = self.block4_4(x4)


        x1 = self.x1_to_240(x1)
        x2 = self.x2_to_240(x2)

        fbm_1_2 = self.fbm_x1_x2((x1,x2))
        fbm_1_3 = self.fbm_x1_x3((x1, x3))
        fbm_1_4 = self.fbm_x1_x4((x1, x4))
        fbm_2_3 = self.fbm_x2_x3((x2, x3))
        fbm_2_4 = self.fbm_x2_x4((x2, x4))
        fbm_3_4 = self.fbm_x3_x4((x3, x4))

        x_fuses = []
        if self.sa and self.dil is not None:
            for i, xi in enumerate([fbm_1_2, fbm_1_3, fbm_1_4, fbm_2_3,fbm_2_4,fbm_3_4]):
                x_fuses.append(self.attentions[i](self.dilations[i](xi)))  # dilations对应CDCM，attentions对应CAM
        elif self.sa:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.attentions[i](xi))
        elif self.dil is not None:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.dilations[i](xi))
        else:
            x_fuses = [x1, x2, x3, x4]


        e1 = self.conv_reduces[0](x_fuses[0])
        e1 = F.interpolate(e1, (H, W), mode="bilinear", align_corners=False)
        e2 = self.conv_reduces[1](x_fuses[1])
        e2 = F.interpolate(e2, (H, W), mode="bilinear", align_corners=False)
        e3 = self.conv_reduces[2](x_fuses[2])
        e3 = F.interpolate(e3, (H, W), mode="bilinear", align_corners=False)
        e4 = self.conv_reduces[3](x_fuses[3])
        e4 = F.interpolate(e4, (H, W), mode="bilinear", align_corners=False)
        e5 = self.conv_reduces[4](x_fuses[4])
        e5 = F.interpolate(e5, (H, W), mode="bilinear", align_corners=False)
        e6 = self.conv_reduces[5](x_fuses[5])
        e6 = F.interpolate(e6, (H, W), mode="bilinear", align_corners=False)


        outputs = [e1, e2, e3, e4,e5,e6]
        output = self.classifier(torch.cat(outputs, dim=1))  # classifier是1*1卷积
        outputs.append(output)
        outputs = [torch.sigmoid(r) for r in outputs]
        return outputs
def ffed(args):
    pdcs = config_model(args.config)  #
    dil = 24 if args.dil else None
    return FFED(60, pdcs, dil=dil, sa=args.sa)


