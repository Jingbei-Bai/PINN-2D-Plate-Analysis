# pinn_model.py - PINN模型定义

import torch
import torch.nn as nn
from config import HIDDEN_LAYERS, NEURONS_PER_LAYER, ACTIVATION


class PINN(nn.Module):
    """物理信息神经网络模型"""

    def __init__(self):
        super(PINN, self).__init__()

        # 输入层: (x, y) -> 隐藏层
        self.input_layer = nn.Linear(2, NEURONS_PER_LAYER)

        # 隐藏层
        self.hidden_layers = nn.ModuleList()
        for _ in range(HIDDEN_LAYERS - 1):
            self.hidden_layers.append(nn.Linear(NEURONS_PER_LAYER, NEURONS_PER_LAYER))

        # 输出层: 隐藏层 -> (u_x, u_y)
        self.output_layer = nn.Linear(NEURONS_PER_LAYER, 2)

        # 激活函数
        self.activation = self.get_activation(ACTIVATION)

    def get_activation(self, name):
        """获取激活函数"""
        if name == 'tanh':
            return torch.tanh
        elif name == 'relu':
            return torch.relu
        elif name == 'sigmoid':
            return torch.sigmoid
        else:
            raise ValueError(f"未知的激活函数: {name}")

    def forward(self, x, y):
        """前向传播"""
        # 合并输入
        inputs = torch.cat([x, y], dim=1)

        # 通过输入层
        out = self.activation(self.input_layer(inputs))

        # 通过隐藏层
        for layer in self.hidden_layers:
            out = self.activation(layer(out))

        # 通过输出层
        outputs = self.output_layer(out)

        return outputs