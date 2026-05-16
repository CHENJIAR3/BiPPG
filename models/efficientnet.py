# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear
# https://github.com/Nokia-Bell-Labs/papagei-foundation-model/blob/main/models/efficientnet.py
import torch 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import torch.nn as nn
from models.net1D import make_mlp_head
class SEModule(torch.nn.Module):
    def __init__(self,in_channel,ratio=4):
        super(SEModule, self).__init__()
        self.avepool = torch.nn.AdaptiveAvgPool1d(1)
        self.linear1 = torch.nn.Linear(in_channel,in_channel//ratio)
        self.linear2 = torch.nn.Linear(in_channel//ratio,in_channel)
        self.Hardsigmoid = torch.nn.Hardsigmoid(inplace=True)
        self.Relu = torch.nn.ReLU(inplace=True)

    def forward(self,input):
        b,c,_ = input.shape
        x = self.avepool(input)
        x = x.view([b,c])
        x = self.linear1(x)
        x = self.Relu(x)
        x = self.linear2(x)
        x = self.Hardsigmoid(x)
        x = x.view([b,c,1])

        return input*x


class MBConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride, se_ratio=4):
        super(MBConvBlock, self).__init__()
        # Expansion phase
        expanded_channels = int(in_channels * expand_ratio)
        self.expand_conv = torch.nn.Conv1d(in_channels, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = torch.nn.BatchNorm1d(expanded_channels)
        # Depthwise convolution
        self.depthwise_conv = torch.nn.Conv1d(expanded_channels, expanded_channels, kernel_size=kernel_size, stride=stride,
                                        padding=kernel_size // 2, groups=expanded_channels, bias=False)
        self.bn2 = torch.nn.BatchNorm1d(expanded_channels)
        # Squeeze and Excitation (SE) phase
        self.se = SEModule(expanded_channels, se_ratio)
        # Linear Bottleneck
        self.linear_bottleneck = torch.nn.Conv1d(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)
        # Skip connection if input and output channels are the same and stride is 1
        self.use_skip_connection = (stride == 1) and (in_channels == out_channels)
        self.leakyrelu = torch.nn.LeakyReLU(0.02)

    def forward(self, x):
        identity = x
        # Expansion phase
        x = self.leakyrelu(self.bn1(self.expand_conv(x)))
        # Depthwise convolution phase
        x = self.leakyrelu(self.bn2(self.depthwise_conv(x)))
        # Squeeze and Excitation phase
        x = self.se(x)
        # Linear Bottleneck phase
        x = self.bn3(self.linear_bottleneck(x))

        # Skip connection
        if self.use_skip_connection:
            x = identity + x

        return x


class EfficientNetB0(torch.nn.Module):
    def __init__(self, args):
        super(EfficientNetB0, self).__init__()

        # Initial stem convolution
        self.stem = torch.nn.Sequential(
            torch.nn.Conv1d(args.in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm1d(32),
            torch.nn.LeakyReLU(0.02)
        )

        # Building blocks
        self.blocks = torch.nn.Sequential(
            MBConvBlock(32, 16, 1, 3, 1),

            MBConvBlock(16, 24, 6, 3, 2),
            MBConvBlock(24, 24, 6, 3, 1),

            MBConvBlock(24, 40, 6, 5, 2),
            MBConvBlock(40, 40, 6, 5, 1),

            MBConvBlock(40, 80, 6, 3, 2),
            MBConvBlock(80, 80, 6, 3, 1),
            MBConvBlock(80, 80, 6, 3, 1),


            MBConvBlock(80, 112, 6, 5, 1),
            MBConvBlock(112, 112, 6, 5, 1),
            MBConvBlock(112, 112, 6, 5, 1),

            MBConvBlock(112, 192, 6, 5, 2),
            MBConvBlock(192, 192, 6, 5, 1),
            MBConvBlock(192, 192, 6, 5, 1),
            # MBConvBlock(192, 192, 6, 5, 1),

            # MBConvBlock(192, 320, 6, 3, 1),
        )

        # Head
        self.conv_head = torch.nn.Sequential(
            torch.nn.Conv1d(192, 1280, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm1d(1280),
            torch.nn.LeakyReLU(0.02)
        )
        
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        
        self.mlp_head = torch.nn.Sequential(
            torch.nn.Linear(1280, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, args.out_dimjj)
        )
        # Global average pooling and classifier
        # 
        # self.fc = torch.nn.Linear(1280, classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp_head(x)
        return x

class EfficientNetB0Small(torch.nn.Module):
    def __init__(self, args):
        super(EfficientNetB0Small, self).__init__()

        # Initial stem convolution
        self.stem = torch.nn.Sequential(
            torch.nn.Conv1d(args.in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm1d(32),
            torch.nn.LeakyReLU(0.02)
        )

        # Building blocks
        self.blocks = torch.nn.Sequential(
            MBConvBlock(32, 16, 1, 3, 1),

            MBConvBlock(16, 24, 6, 3, 2),
            MBConvBlock(24, 24, 6, 3, 1),

            # MBConvBlock(24, 40, 6, 5, 2),
            # MBConvBlock(40, 40, 6, 5, 1),

            # MBConvBlock(40, 80, 6, 3, 2),
            # MBConvBlock(80, 80, 6, 3, 1),
            # MBConvBlock(80, 80, 6, 3, 1),


            # MBConvBlock(80, 112, 6, 5, 1),
            # MBConvBlock(112, 112, 6, 5, 1),
            # MBConvBlock(112, 112, 6, 5, 1),

            # MBConvBlock(112, 192, 6, 5, 2),
            # MBConvBlock(192, 192, 6, 5, 1),
            # MBConvBlock(192, 192, 6, 5, 1),
            # MBConvBlock(192, 192, 6, 5, 1),

            # MBConvBlock(192, 320, 6, 3, 1),
        )

        # Head
        # self.conv_head = torch.nn.Sequential(
        #     torch.nn.Conv1d(112, 1280, kernel_size=1, stride=1, padding=0, bias=False),
        #     torch.nn.BatchNorm1d(1280),
        #     torch.nn.LeakyReLU(0.02)
        # )
        
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        
        self.mlp_head = torch.nn.Sequential(
            torch.nn.Linear(24, args.out_dim),
            # torch.nn.ReLU(),
            # torch.nn.Linear(512, 256),
            # torch.nn.ReLU(),
            # torch.nn.Linear(256, 128)            
        )
        # Global average pooling and classifier
        # 
        # self.fc = torch.nn.Linear(1280, classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        # x = self.conv_head(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        # x = x.squeeze()
        x = self.mlp_head(x)
        return x

class EfficientNetB0Base(torch.nn.Module):
    def __init__(self, in_channels, dict_channels):
        super(EfficientNetB0Base, self).__init__()

        # Initial stem convolution
        self.stem = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, dict_channels['h1'], kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm1d(dict_channels['h1']),
            torch.nn.LeakyReLU(0.02)
        )

        # Building blocks
        self.blocks = torch.nn.Sequential(
            MBConvBlock(dict_channels['h1'], dict_channels['h1'] // 2, 1, 3, 1),

            MBConvBlock(dict_channels['h1'] // 2, dict_channels['h2'], 6, 3, 2),
            MBConvBlock(dict_channels['h2'], dict_channels['h2'], 6, 3, 1),

            MBConvBlock(dict_channels['h2'], dict_channels['h3'], 6, 5, 2),
            MBConvBlock(dict_channels['h3'], dict_channels['h3'], 6, 5, 1),

            MBConvBlock(dict_channels['h3'], dict_channels['h4'], 6, 3, 2),
            MBConvBlock(dict_channels['h4'], dict_channels['h4'], 6, 3, 1),
            MBConvBlock(dict_channels['h4'], dict_channels['h4'], 6, 3, 1),


            MBConvBlock(dict_channels['h4'], dict_channels['h5'], 6, 5, 1),
            MBConvBlock(dict_channels['h5'], dict_channels['h5'], 6, 5, 1),
            MBConvBlock(dict_channels['h5'], dict_channels['h5'], 6, 5, 1),

            MBConvBlock(dict_channels['h5'], dict_channels['h6'], 6, 5, 2),
            MBConvBlock(dict_channels['h6'], dict_channels['h6'], 6, 5, 1),
            MBConvBlock(dict_channels['h6'], dict_channels['h6'], 6, 5, 1),
            MBConvBlock(dict_channels['h6'], dict_channels['h6'], 6, 5, 1),

            MBConvBlock(dict_channels['h6'], dict_channels['h7'], 6, 3, 1),
        )

        self.conv_head = torch.nn.Sequential(
            torch.nn.Conv1d(dict_channels['h7'], dict_channels['h8'], kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm1d(dict_channels['h8']),
            torch.nn.LeakyReLU(0.02)
        )
        
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        
        self.mlp_head = torch.nn.Sequential(
            torch.nn.Linear(dict_channels['h8'], 1024),
            # torch.nn.ReLU(),
            # torch.nn.Linear(1024, dict_channels["out_dim"]),
        )
        self.heads = nn.ModuleDict({
            "sbp": make_mlp_head(1024, 1),  # SBP
            "dbp": make_mlp_head(1024, 1),  # DBP
            "hr": make_mlp_head(1024, 1),  # HR
            "age": make_mlp_head(1024, 1),  # Age

        })

        # Global average pooling and classifier
        # 
        # self.fc = torch.nn.Linear(1280, classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        # x = x.squeeze()
        x = self.mlp_head(x)
        outputs = {
            "sbp": self.heads["sbp"](x),  # (N, 2)
            "dbp": self.heads["dbp"](x),  # (N, 2)
            "hr": self.heads["hr"](x),  # (N, 1)
            "age": self.heads["age"](x),  # (N, 1)

        }
        values = [
            outputs["sbp"],
            outputs["dbp"],
            outputs["hr"],
            outputs["age"],

        ]
        outputs_np = torch.cat(values,dim=1)
        return outputs_np

def myEfficientNet(args):
    model_config = {'h1': 64,
                    'h2': 32,
                    'h3': 128,
                    'h4': 256,
                    'h5': 384,
                    'h6': 512,
                    'h7': 768,
                    'h8': 1024,
                    "out_dim":args.out_dim,}

    return EfficientNetB0Base(in_channels=args.in_channels, dict_channels=model_config)
if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    from utils import  args,get_total_params
    model =  myEfficientNet(args)
    x = torch.randn((1,4,1000))
    # print(model(x)["sbp"].shape)
    # 3. 计算 FLOPs
    flops = FlopCountAnalysis(model, x)
    print(f"Total FLOPs: {flops.total() / 1e6:.2f} M")  # 以百万为单位
    total_params = get_total_params(model)
    print(f"模型参数总数: {total_params:,}")
    # 4. 计算并打印参数表格 (非常适合放进 PPT 或论文附录)
    # print(parameter_count_table(model))
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(
        model,
        (4, 1000),
        as_strings=True,
        print_per_layer_stat=True
    )
    print('MACs:', macs)
    print('Params:', params)