# PINN-2D-Plate-Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个基于物理信息神经网络（PINN）的二维平板力学分析工具，支持多种边界条件配置，用于解决固体力学中的弹性变形问题。（遗憾的是，本项目目前并不成熟，我们预计会在未来的三个月内持续更新并融入更加强大的机器学习组件。同时也感谢所有看到/使用本项目的同僚为此项目纠错）

## 项目简介

PINN-2D-Plate-Analysis 是一个创新的计算力学软件，它将深度学习的强大能力与传统的固体力学原理相结合。该项目利用物理信息神经网络（PINN）技术，为二维弹性体的力学分析提供了一种全新的解决方案。

### 技术背景

传统的有限元方法（FEM）在解决复杂边界条件问题时需要精细的网格划分和大量的计算资源。PINN方法通过将物理方程直接嵌入神经网络的训练过程中，实现了无需网格划分的高精度计算，特别适合处理复杂边界条件和逆问题。

### 核心创新

1. **多边界条件支持**: 首次在PINN框架中实现了固定、自由、函数和插值四种边界条件的统一处理
2. **硬约束技术**: 通过距离函数确保边界条件被严格满足，提高了计算精度
3. **自适应损失平衡**: 动态调整PDE残差和边界条件损失的权重，优化训练过程
4. **全面可视化**: 提供从位移场到应力场的完整力学分析可视化

## 安装指南

### 环境要求

- Python 3.8+
- PyTorch 1.9+
- CUDA（可选，用于GPU加速）

### 安装步骤

1. 克隆项目
git clone https://github.com/Jingbei-Bai/PINN-2D-Plate-Analysis.git
cd PINN-2D-Plate-Analysis
2. 安装依赖
pip install -r requirements.txt
3. 运行示例
python main.py
## 使用说明

### 快速开始

1. **配置参数**：修改 `config.py` `boundary_conditions.py`中的参数
2. **运行训练**：执行 `python main.py`
3. **查看结果**：程序自动生成可视化图像

### 配置条件

在 `config.py` 中配置如训练率、边界条件等等超参数


### 边界条件类型

- **固定边界**：位移为零的约束边界
- **自由边界**：无约束，允许自由变形
- **函数边界**：通过数学函数定义边界位移
- **插值边界**：通过插值点定义复杂边界位移

## 项目结构

PINN-2D-Plate-Analysis/

├── main.py # 主训练程序

├── pinn_model.py # PINN神经网络模型

├── physics.py # 物理方程和硬约束

├── boundary_conditions.py # 边界条件定义

├── config.py # 配置参数

├── utils.py # 辅助函数和绘图

├── boundary_visualization.py # 边界可视化

├── requirements.txt # 项目依赖

└── README.md # 项目说明

## 结果展示

默认配置输出示例

运行默认配置（左侧固定，右侧插值边界，上侧自由，下侧自由）将生成以下结果：

1. 位移场分布

<img width="1200" height="500" alt="displacement_field" src="https://github.com/user-attachments/assets/953d8d31-18ef-44b8-9c73-8b7cdf0de52e" />

图1: x方向和y方向的位移分布

3. 应力应变场

<img width="1500" height="1000" alt="stress_strain_field" src="https://github.com/user-attachments/assets/1d16ea97-11a4-4df9-bd85-7f77dc315e39" />

图2: 应变分量和应力分量分布

5. Von Mises应力分布

<img width="800" height="600" alt="von_mises_stress" src="https://github.com/user-attachments/assets/29d8988e-1a6d-4d2e-b3fb-ed0e06b094a4" />

图3: Von Mises等效应力分布

7. 边界位移分析

<img width="4467" height="2968" alt="boundary_displacement_overview" src="https://github.com/user-attachments/assets/e1a2eb99-b016-43c0-b6d9-5849dd25ccba" />

图4: 边界位移详细分析

## 贡献指南

我们欢迎各种形式的贡献！请参考以下步骤：

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 联系方式

- 项目主页：https://github.com/Jingbei-Bai/PINN-2D-Plate-Analysis
- 问题反馈：https://github.com/Jingbei-Bai/PINN-2D-Plate-Analysis/issues
- 邮箱：bjb23@mails.tsinghua.edu.cn

