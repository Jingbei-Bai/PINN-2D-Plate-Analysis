# boundary_conditions.py - 边界条件定义

import numpy as np
import torch
from scipy.interpolate import interp1d
from config import PLATE_WIDTH, PLATE_HEIGHT, INTERPOLATION_POINTS

# 边界类型常量
BOUNDARY_FUNCTION = 'function'
BOUNDARY_INTERPOLATION = 'interpolation'
BOUNDARY_FREE = 'free'
BOUNDARY_FIXED = 'fixed'


def boundary_condition_function(x, y):
    """
    使用函数定义边界条件

    参数:
        x, y: 坐标张量

    返回:
        u_x_bc, u_y_bc: 边界位移分量
    """
    # 初始化边界位移
    u_x_bc = torch.zeros_like(x)
    u_y_bc = torch.zeros_like(y)

    # 右侧边界 (x=PLATE_WIDTH): 正弦位移
    right_mask = (x >= PLATE_WIDTH - 1e-5)
    u_x_bc[right_mask] = 0.001 * torch.sin(2 * torch.pi * 2 * y[right_mask] / PLATE_HEIGHT)

    return u_x_bc, u_y_bc


def boundary_condition_interpolation(x, y):
    """
    使用插值定义边界条件

    参数:
        x, y: 坐标张量

    返回:
        u_x_bc, u_y_bc: 边界位移分量
    """
    # 初始化边界位移
    u_x_bc = torch.zeros_like(x)
    u_y_bc = torch.zeros_like(y)

    # 右侧边界 (x=PLATE_WIDTH): 使用插值
    right_mask = (x >= PLATE_WIDTH - 1e-5)

    # 如果存在右侧边界点
    if torch.any(right_mask):
        # 将张量移动到CPU并转换为NumPy数组
        y_vals = y[right_mask].cpu().detach().numpy().flatten()  # 确保是一维数组

        # 创建插值函数
        y_points = np.linspace(0, PLATE_HEIGHT, INTERPOLATION_POINTS)
        u_x_points = np.array([0.000, 0.001, 0.0005, -0.001, -0.0005, 0.0008, 0.0012, 0.0005, -0.0007, 0.0])
        interp_func = interp1d(y_points, u_x_points, kind='cubic', fill_value="extrapolate")

        # 应用插值
        u_x_vals = interp_func(y_vals)

        # 将结果移回原设备
        u_x_bc[right_mask] = torch.tensor(u_x_vals, dtype=torch.float32, device=x.device)

    return u_x_bc, u_y_bc


def boundary_condition_free(x, y):
    """
    自由边界条件（无约束）

    参数:
        x, y: 坐标张量

    返回:
        u_x_bc, u_y_bc: 边界位移分量（全零）
    """
    return torch.zeros_like(x), torch.zeros_like(y)


def boundary_condition_fixed(x, y):
    """
    固定边界条件（位移为零）

    参数:
        x, y: 坐标张量

    返回:
        u_x_bc, u_y_bc: 边界位移分量（全零）
    """
    return torch.zeros_like(x), torch.zeros_like(y)


def get_boundary_condition(boundary_type):
    """
    获取边界条件函数

    参数:
        boundary_type: 'function', 'interpolation', 'free', 'fixed'

    返回:
        边界条件函数
    """
    if boundary_type == BOUNDARY_FUNCTION:
        return boundary_condition_function
    elif boundary_type == BOUNDARY_INTERPOLATION:
        return boundary_condition_interpolation
    elif boundary_type == BOUNDARY_FREE:
        return boundary_condition_free
    elif boundary_type == BOUNDARY_FIXED:
        return boundary_condition_fixed
    else:
        raise ValueError(f"未知的边界类型: {boundary_type}")