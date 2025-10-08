# physics.py - 物理方程和边界处理

import torch
from config import YOUNG_MODULUS, POISSON_RATIO, PLATE_WIDTH, PLATE_HEIGHT, BOUNDARY_LEFT, BOUNDARY_RIGHT, BOUNDARY_TOP, \
    BOUNDARY_BOTTOM


def constitutive_relation(strain):
    """
    线弹性材料的本构关系 (平面应力)
    返回归一化应力 (除以杨氏模量)
    """
    # 平面应力问题的弹性矩阵
    C11 = 1 / (1 - POISSON_RATIO ** 2)  # 归一化后的系数
    C12 = POISSON_RATIO * C11
    C33 = 1 / (2 * (1 + POISSON_RATIO))

    # 应力-应变关系 (归一化)
    σ_xx = C11 * strain[:, 0:1] + C12 * strain[:, 1:2]
    σ_yy = C12 * strain[:, 0:1] + C11 * strain[:, 1:2]
    τ_xy = C33 * strain[:, 2:3]

    return torch.cat([σ_xx, σ_yy, τ_xy], dim=1)


def von_mises_stress(stress_norm):
    """
    计算归一化Von Mises应力 (平面应力)
    参数: stress_norm - 归一化应力张量 [σ_xx_norm, σ_yy_norm, τ_xy_norm]
    返回: 归一化Von Mises应力
    """
    σ_xx = stress_norm[:, 0:1]
    σ_yy = stress_norm[:, 1:2]
    τ_xy = stress_norm[:, 2:3]

    # Von Mises公式 (平面应力)
    von_mises = torch.sqrt(σ_xx ** 2 - σ_xx * σ_yy + σ_yy ** 2 + 3 * τ_xy ** 2)
    return von_mises


def strain_displacement(u, x, y):
    """计算应变 (小变形理论)"""
    u_x = u[:, 0:1]
    u_y = u[:, 1:2]

    # 确保所有张量都参与了计算图
    u_x = u_x + 0 * x
    u_y = u_y + 0 * y

    # 计算位移梯度
    dux_dx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                 create_graph=True, retain_graph=True, allow_unused=True)[0]
    duy_dy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y),
                                 create_graph=True, retain_graph=True, allow_unused=True)[0]
    dux_dy = torch.autograd.grad(u_x, y, grad_outputs=torch.ones_like(u_x),
                                 create_graph=True, retain_graph=True, allow_unused=True)[0]
    duy_dx = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y),
                                 create_graph=True, retain_graph=True, allow_unused=True)[0]

    # 处理未使用的梯度（设置为0）
    if dux_dx is None:
        dux_dx = torch.zeros_like(u_x)
    if duy_dy is None:
        duy_dy = torch.zeros_like(u_y)
    if dux_dy is None:
        dux_dy = torch.zeros_like(u_x)
    if duy_dx is None:
        duy_dx = torch.zeros_like(u_y)

    # 计算应变分量
    ε_xx = dux_dx
    ε_yy = duy_dy
    γ_xy = dux_dy + duy_dx

    return torch.cat([ε_xx, ε_yy, γ_xy], dim=1)


def equilibrium_equations(u, x, y):
    """
    平衡方程 (无体力情况)
    使用归一化应力
    """
    # 计算应变
    strain = strain_displacement(u, x, y)

    # 计算归一化应力
    stress_norm = constitutive_relation(strain)
    σ_xx_norm = stress_norm[:, 0:1]
    σ_yy_norm = stress_norm[:, 1:2]
    τ_xy_norm = stress_norm[:, 2:3]

    # 计算应力梯度
    dσxx_dx = torch.autograd.grad(σ_xx_norm, x, grad_outputs=torch.ones_like(σ_xx_norm),
                                  create_graph=True, retain_graph=True, allow_unused=True)[0]
    dτxy_dy = torch.autograd.grad(τ_xy_norm, y, grad_outputs=torch.ones_like(τ_xy_norm),
                                  create_graph=True, retain_graph=True, allow_unused=True)[0]
    dτxy_dx = torch.autograd.grad(τ_xy_norm, x, grad_outputs=torch.ones_like(τ_xy_norm),
                                  create_graph=True, retain_graph=True, allow_unused=True)[0]
    dσyy_dy = torch.autograd.grad(σ_yy_norm, y, grad_outputs=torch.ones_like(σ_yy_norm),
                                  create_graph=True, retain_graph=True, allow_unused=True)[0]

    # 处理未使用的梯度（设置为0）
    if dσxx_dx is None:
        dσxx_dx = torch.zeros_like(σ_xx_norm)
    if dτxy_dy is None:
        dτxy_dy = torch.zeros_like(τ_xy_norm)
    if dτxy_dx is None:
        dτxy_dx = torch.zeros_like(τ_xy_norm)
    if dσyy_dy is None:
        dσyy_dy = torch.zeros_like(σ_yy_norm)

    # 平衡方程残差
    res_x = dσxx_dx + dτxy_dy
    res_y = dτxy_dx + dσyy_dy

    return res_x, res_y


def hard_constraint_displacement(x, y, net_output,
                                 boundary_func_left, boundary_func_right,
                                 boundary_func_top, boundary_func_bottom):
    """
    硬约束位移场（修正自由边界处理）

    参数:
        x, y: 坐标位置
        net_output: 神经网络的原始输出 [u_x_raw, u_y_raw]
        boundary_func_left: 左侧边界函数
        boundary_func_right: 右侧边界函数
        boundary_func_top: 上侧边界函数
        boundary_func_bottom: 下侧边界函数

    返回:
        u: 位移向量 [u_x, u_y]
    """
    u_x_raw = net_output[:, 0:1]
    u_y_raw = net_output[:, 1:2]

    # 初始化混合函数（内部点为1，边界点为0）
    blend = torch.ones_like(x)

    # 左侧边界 (x=0)
    dist_left = x / PLATE_WIDTH
    if BOUNDARY_LEFT != 'free':  # 非自由边界才应用约束
        blend = blend * dist_left

    # 右侧边界 (x=PLATE_WIDTH)
    dist_right = (PLATE_WIDTH - x) / PLATE_WIDTH
    if BOUNDARY_RIGHT != 'free':  # 非自由边界才应用约束
        blend = blend * dist_right

    # 上边界 (y=PLATE_HEIGHT)
    dist_top = (PLATE_HEIGHT - y) / PLATE_HEIGHT
    if BOUNDARY_TOP != 'free':  # 非自由边界才应用约束
        blend = blend * dist_top

    # 下边界 (y=0)
    dist_bottom = y / PLATE_HEIGHT
    if BOUNDARY_BOTTOM != 'free':  # 非自由边界才应用约束
        blend = blend * dist_bottom

    # 获取边界条件
    u_x_bc_left, u_y_bc_left = boundary_func_left(x, y)
    u_x_bc_right, u_y_bc_right = boundary_func_right(x, y)
    u_x_bc_top, u_y_bc_top = boundary_func_top(x, y)
    u_x_bc_bottom, u_y_bc_bottom = boundary_func_bottom(x, y)

    # 合并边界条件
    u_x_bc = u_x_bc_left + u_x_bc_right + u_x_bc_top + u_x_bc_bottom
    u_y_bc = u_y_bc_left + u_y_bc_right + u_y_bc_top + u_y_bc_bottom

    # 应用硬约束
    u_x = blend * u_x_raw + u_x_bc
    u_y = blend * u_y_raw + u_y_bc

    return torch.cat([u_x, u_y], dim=1)