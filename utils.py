# utils.py - 辅助函数

import numpy as np
import torch
import matplotlib.pyplot as plt
from config import PLATE_WIDTH, PLATE_HEIGHT, YOUNG_MODULUS
import physics
from pinn_model import PINN
from boundary_conditions import get_boundary_condition


def generate_domain_points(nx=100, ny=100, device='cpu'):
    """生成域内点并移动到指定设备"""
    x = np.linspace(0, PLATE_WIDTH, nx)
    y = np.linspace(0, PLATE_HEIGHT, ny)
    X, Y = np.meshgrid(x, y)
    X_flat = X.reshape(-1, 1)
    Y_flat = Y.reshape(-1, 1)

    # 转换为PyTorch张量
    x_tensor = torch.tensor(X_flat, dtype=torch.float32, requires_grad=True).to(device)
    y_tensor = torch.tensor(Y_flat, dtype=torch.float32, requires_grad=True).to(device)

    return x_tensor, y_tensor


def plot_results(model, boundary_func_left, boundary_func_right, boundary_func_top, boundary_func_bottom, nx=50, ny=50):
    """
    绘制结果（支持四个边界）

    参数:
        model: 训练好的PINN模型
        boundary_func_left: 左侧边界函数
        boundary_func_right: 右侧边界函数
        boundary_func_top: 上侧边界函数
        boundary_func_bottom: 下侧边界函数
        nx, ny: 网格点数
    """
    # 生成网格点
    x = np.linspace(0, PLATE_WIDTH, nx)
    y = np.linspace(0, PLATE_HEIGHT, ny)
    X, Y = np.meshgrid(x, y)

    # 转换为PyTorch张量
    x_tensor = torch.tensor(X.reshape(-1, 1), dtype=torch.float32)
    y_tensor = torch.tensor(Y.reshape(-1, 1), dtype=torch.float32)

    # 预测位移
    with torch.no_grad():
        net_out = model(x_tensor, y_tensor)
        u_pred = physics.hard_constraint_displacement(x_tensor, y_tensor, net_out,
                                                      boundary_func_left, boundary_func_right,
                                                      boundary_func_top, boundary_func_bottom)
        u_x = u_pred[:, 0].numpy().reshape(nx, ny)
        u_y = u_pred[:, 1].numpy().reshape(nx, ny)

    # 绘制位移场
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.contourf(X, Y, u_x, 20, cmap='viridis')
    plt.colorbar(label='u_x (m)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Displacement in x-direction')

    plt.subplot(1, 2, 2)
    plt.contourf(X, Y, u_y, 20, cmap='viridis')
    plt.colorbar(label='u_y (m)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Displacement in y-direction')

    plt.tight_layout()
    plt.savefig('displacement_field.png')
    plt.close()

    # 计算应力和应变
    x_tensor.requires_grad = True
    y_tensor.requires_grad = True

    net_out = model(x_tensor, y_tensor)
    u_pred = physics.hard_constraint_displacement(x_tensor, y_tensor, net_out,
                                                  boundary_func_left, boundary_func_right,
                                                  boundary_func_top, boundary_func_bottom)
    strain = physics.strain_displacement(u_pred, x_tensor, y_tensor)
    stress_norm = physics.constitutive_relation(strain)

    # 提取应变分量
    ε_xx = strain[:, 0].detach().numpy().reshape(nx, ny)
    ε_yy = strain[:, 1].detach().numpy().reshape(nx, ny)
    γ_xy = strain[:, 2].detach().numpy().reshape(nx, ny)

    # 提取归一化应力分量并转换为实际应力
    σ_xx_norm = stress_norm[:, 0].detach().numpy().reshape(nx, ny)
    σ_yy_norm = stress_norm[:, 1].detach().numpy().reshape(nx, ny)
    τ_xy_norm = stress_norm[:, 2].detach().numpy().reshape(nx, ny)

    σ_xx = σ_xx_norm * YOUNG_MODULUS
    σ_yy = σ_yy_norm * YOUNG_MODULUS
    τ_xy = τ_xy_norm * YOUNG_MODULUS

    # 计算Von Mises应力 (实际)
    von_mises_norm = physics.von_mises_stress(stress_norm)
    von_mises = von_mises_norm.detach().numpy().reshape(nx, ny) * YOUNG_MODULUS

    # 绘制应变场
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.contourf(X, Y, ε_xx, 20, cmap='jet')
    plt.colorbar(label='ε_xx')
    plt.title('Normal Strain in x-direction')

    plt.subplot(2, 3, 2)
    plt.contourf(X, Y, ε_yy, 20, cmap='jet')
    plt.colorbar(label='ε_yy')
    plt.title('Normal Strain in y-direction')

    plt.subplot(2, 3, 3)
    plt.contourf(X, Y, γ_xy, 20, cmap='jet')
    plt.colorbar(label='γ_xy')
    plt.title('Shear Strain')

    # 绘制应力场
    plt.subplot(2, 3, 4)
    plt.contourf(X, Y, σ_xx, 20, cmap='jet')
    plt.colorbar(label='σ_xx (Pa)')
    plt.title('Normal Stress in x-direction')

    plt.subplot(2, 3, 5)
    plt.contourf(X, Y, σ_yy, 20, cmap='jet')
    plt.colorbar(label='σ_yy (Pa)')
    plt.title('Normal Stress in y-direction')

    plt.subplot(2, 3, 6)
    plt.contourf(X, Y, τ_xy, 20, cmap='jet')
    plt.colorbar(label='τ_xy (Pa)')
    plt.title('Shear Stress')

    plt.tight_layout()
    plt.savefig('stress_strain_field.png')
    plt.close()

    # 绘制Von Mises应力
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, von_mises, 20, cmap='jet')
    plt.colorbar(label='Von Mises Stress (Pa)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Von Mises Stress Distribution')
    plt.savefig('von_mises_stress.png')
    plt.close()

    # 计算和打印PDE残差和边界误差
    # PDE残差在整个域上
    res_x, res_y = physics.equilibrium_equations(u_pred, x_tensor, y_tensor)
    pde_residual = torch.mean(res_x ** 2 + res_y ** 2).item()
    print(f"Mean PDE residual: {pde_residual:.6e}")

    # 边界误差（即使有硬约束，我们也可以检查数值误差）
    # 左侧边界 (x=0): 应为0位移
    left_mask = (x_tensor <= 1e-5).squeeze()
    if left_mask.any():
        u_left = u_pred[left_mask]
        left_error = torch.mean(u_left ** 2).item()
        print(f"Left boundary error (MSE): {left_error:.6e}")
    else:
        print("左侧边界无点")

    # 右侧边界 (x=PLATE_WIDTH)
    right_mask = (x_tensor >= PLATE_WIDTH - 1e-5).squeeze()
    if right_mask.any():
        u_right = u_pred[right_mask]
        u_x_right = u_right[:, 0:1]
        u_y_right = u_right[:, 1:2]

        # 目标值
        u_x_bc, u_y_bc = boundary_func_right(x_tensor[right_mask], y_tensor[right_mask])

        # 边界条件损失
        right_error_x = torch.mean((u_x_right - u_x_bc) ** 2).item()
        right_error_y = torch.mean((u_y_right - u_y_bc) ** 2).item()
        total_right_error = right_error_x + right_error_y
        print(f"Right boundary error - X (MSE): {right_error_x:.6e}")
        print(f"Right boundary error - Y (MSE): {right_error_y:.6e}")
        print(f"Total right boundary error (MSE): {total_right_error:.6e}")
    else:
        print("右侧边界无点")

    # 上边界 (y=PLATE_HEIGHT)
    top_mask = (y_tensor >= PLATE_HEIGHT - 1e-5).squeeze()
    if top_mask.any():
        u_top = u_pred[top_mask]
        u_x_top = u_top[:, 0:1]
        u_y_top = u_top[:, 1:2]

        # 目标值
        u_x_bc, u_y_bc = boundary_func_top(x_tensor[top_mask], y_tensor[top_mask])

        # 边界条件损失
        top_error_x = torch.mean((u_x_top - u_x_bc) ** 2).item()
        top_error_y = torch.mean((u_y_top - u_y_bc) ** 2).item()
        total_top_error = top_error_x + top_error_y
        print(f"Top boundary error - X (MSE): {top_error_x:.6e}")
        print(f"Top boundary error - Y (MSE): {top_error_y:.6e}")
        print(f"Total top boundary error (MSE): {total_top_error:.6e}")
    else:
        print("上边界无点")

    # 下边界 (y=0)
    bottom_mask = (y_tensor <= 1e-5).squeeze()
    if bottom_mask.any():
        u_bottom = u_pred[bottom_mask]
        u_x_bottom = u_bottom[:, 0:1]
        u_y_bottom = u_bottom[:, 1:2]

        # 目标值
        u_x_bc, u_y_bc = boundary_func_bottom(x_tensor[bottom_mask], y_tensor[bottom_mask])

        # 边界条件损失
        bottom_error_x = torch.mean((u_x_bottom - u_x_bc) ** 2).item()
        bottom_error_y = torch.mean((u_y_bottom - u_y_bc) ** 2).item()
        total_bottom_error = bottom_error_x + bottom_error_y
        print(f"Bottom boundary error - X (MSE): {bottom_error_x:.6e}")
        print(f"Bottom boundary error - Y (MSE): {bottom_error_y:.6e}")
        print(f"Total bottom boundary error (MSE): {total_bottom_error:.6e}")
    else:
        print("下边界无点")