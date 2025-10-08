# boundary_visualization.py - 边界可视化

import numpy as np
import torch
import matplotlib.pyplot as plt
from config import PLATE_WIDTH, PLATE_HEIGHT, BOUNDARY_LEFT, BOUNDARY_RIGHT, BOUNDARY_TOP, BOUNDARY_BOTTOM
import physics
from pinn_model import PINN
from boundary_conditions import get_boundary_condition


def plot_boundary_displacement(model, constraint_func,
                               boundary_func_left, boundary_func_right,
                               boundary_func_top, boundary_func_bottom,
                               scale_factor=1000.0):
    """
    绘制边界位移预测结果

    参数:
        model: 训练好的PINN模型
        constraint_func: 硬约束函数
        boundary_func_left: 左侧边界函数
        boundary_func_right: 右侧边界函数
        boundary_func_top: 上侧边界函数
        boundary_func_bottom: 下侧边界函数
        scale_factor: 位移放大因子
    """
    # 高密度采样边界点
    n_points = 200

    # 创建边界点
    # 左侧边界 (x=0)
    x_left = np.zeros(n_points)
    y_left = np.linspace(0, PLATE_HEIGHT, n_points)

    # 右侧边界 (x=PLATE_WIDTH)
    x_right = np.ones(n_points) * PLATE_WIDTH
    y_right = np.linspace(0, PLATE_HEIGHT, n_points)

    # 上边界 (y=PLATE_HEIGHT)
    x_top = np.linspace(0, PLATE_WIDTH, n_points)
    y_top = np.ones(n_points) * PLATE_HEIGHT

    # 下边界 (极y=0)
    x_bottom = np.linspace(0, PLATE_WIDTH, n_points)
    y_bottom = np.zeros(n_points)

    # 合并所有边界点
    x_boundary = np.concatenate([x_left, x_right, x_top, x_bottom])
    y_boundary = np.concatenate([y_left, y_right, y_top, y_bottom])

    # 转换为PyTorch张量
    x_tensor = torch.tensor(x_boundary.reshape(-1, 1), dtype=torch.float32)
    y_tensor = torch.tensor(y_boundary.reshape(-1, 1), dtype=torch.float32)

    # 预测位移
    with torch.no_grad():
        net_out = model(x_tensor, y_tensor)
        u_pred = constraint_func(x_tensor, y_tensor, net_out,
                                 boundary_func_left, boundary_func_right,
                                 boundary_func_top, boundary_func_bottom)
        u_x = u_pred[:, 0].numpy().flatten()  # 确保是一维数组
        u_y = u_pred[:, 1].numpy().flatten()  # 确保是一维数组

    # 计算变形后的位置
    x_deformed = x_boundary + u_x * scale_factor
    y_deformed = y_boundary + u_y * scale_factor

    # 创建绘图
    plt.figure(figsize=(15, 10))

    # 1. 原始边界框架
    plt.subplot(2, 2, 1)
    plt.plot([0, PLATE_WIDTH, PLATE_WIDTH, 0, 0],
             [0, 0, PLATE_HEIGHT, PLATE_HEIGHT, 0], 'k-', linewidth=3)
    plt.title('Original Plate Boundary')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    # 2. 变形后的边界
    plt.subplot(2, 2, 2)
    # 原始边界
    plt.plot([0, PLATE_WIDTH, PLATE_WIDTH, 0, 0],
             [0, 0, PLATE_HEIGHT, PLATE_HEIGHT, 0], 'k--', linewidth=2, label='Original')

    # 变形后的边界点
    plt.scatter(x_deformed, y_deformed, s=20, c='red', label='Deformed Boundary Points')

    # 连接原始位置和变形后位置
    for i in range(len(x_boundary)):
        plt.plot([x_boundary[i], x_deformed[i]],
                 [y_boundary[i], y_deformed[i]],
                 'b-', linewidth=0.5, alpha=0.3)

    plt.title(f'Deformed Boundary (Scale: {scale_factor}x)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    # 3. 边界位移分量
    plt.subplot(2, 2, 3)
    # 左侧边界位移
    plt.plot(y_left, u_x[:n_points], 'b-', label='Left u_x')
    plt.plot(y_left, u_y[:n_points], 'g-', label='Left u_y')

    # 右侧边界位移
    plt.plot(y_right, u_x[n_points:2 * n_points], 'b--', label='Right u_x')
    plt.plot(y_right, u_y[n_points:2 * n_points], 'g--', label='Right u_y')

    plt.xlabel('y (m)')
    plt.ylabel('Displacement (m)')
    plt.title('Left and Right Boundary Displacement')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. 上下边界位移
    plt.subplot(2, 2, 4)
    # 上边界位移
    plt.plot(x_top, u_x[2 * n_points:3 * n_points], 'b-', label='Top u_x')
    plt.plot(x_top, u_y[2 * n_points:3 * n_points], 'g-', label='Top u_y')

    # 下边界位移
    plt.plot(x_bottom, u_x[3 * n_points:], 'b--', label='Bottom u_x')
    plt.plot(x_bottom, u_y[3 * n_points:], 'g--', label='Bottom u_y')

    plt.xlabel('x (m)')
    plt.ylabel('Displacement (m)')
    plt.title('Top and Bottom Boundary Displacement')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('boundary_displacement_overview.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 打印位移统计信息
    print("边界位移统计:")
    print(f"  最大x方向位移: {u_x.max():.6f} m")
    print(f"  最大y方向位移: {u_y.max():.6f} m")
    print(f"  平均位移幅度: {np.sqrt(u_x ** 2 + u_y ** 2).mean():.6f} m")
    print(f"  使用的放大因子: {scale_factor}x")

    print("边界位移图已保存为 'boundary_displacement_overview.png'")


def main():
    """主函数"""
    try:
        # 加载训练好的模型
        model = PINN()
        model.load_state_dict(torch.load('pinn_model.pth'))
        model.eval()
        print("模型加载成功")

        # 获取边界条件函数
        boundary_func_left = get_boundary_condition(BOUNDARY_LEFT)
        boundary_func_right = get_boundary_condition(BOUNDARY_RIGHT)
        boundary_func_top = get_boundary_condition(BOUNDARY_TOP)
        boundary_func_bottom = get_boundary_condition(BOUNDARY_BOTTOM)

        print(f"边界条件配置:")
        print(f"  左侧: {BOUNDARY_LEFT}")
        print(f"  右侧: {BOUNDARY_RIGHT}")
        print(f"  上侧: {BOUNDARY_TOP}")
        print(f"  下侧: {BOUNDARY_BOTTOM}")

        # 绘制边界位移预测结果
        plot_boundary_displacement(model, physics.hard_constraint_displacement,
                                   boundary_func_left, boundary_func_right,
                                   boundary_func_top, boundary_func_bottom,
                                   scale_factor=100.0)

    except FileNotFoundError:
        print("错误: 未找到模型文件 'pinn_model.pth'")
        print("请先运行主训练程序")
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()