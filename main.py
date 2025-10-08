# main.py - 主程序

import torch
import torch.optim as optim
import numpy as np
from pinn_model import PINN
import physics
from utils import generate_domain_points, plot_results
from config import EPOCHS, BATCH_SIZE, LEARNING_RATE, ADAPTIVE_LOSS, BOUNDARY_LEFT, BOUNDARY_RIGHT, BOUNDARY_TOP, \
    BOUNDARY_BOTTOM,PLATE_WIDTH, PLATE_HEIGHT
from boundary_conditions import get_boundary_condition


class AdaptiveLossWeights:
    """自适应损失权重管理"""

    def __init__(self, initial_weights, update_interval=100, smoothing_factor=0.9):
        self.weights = initial_weights.copy()
        self.loss_history = {key: [] for key in initial_weights}
        self.update_interval = update_interval
        self.smoothing_factor = smoothing_factor
        self.update_counter = 0
        self.ema_losses = {key: 0.0 for key in initial_weights}
        self.eps = 1e-8

    def update(self, losses):
        """更新权重"""
        self.update_counter += 1

        # 更新指数移动平均损失
        for key in losses:
            current_loss = losses[key].item()
            self.loss_history[key].append(current_loss)

            # 指数移动平均
            if self.update_counter == 1:
                self.ema_losses[key] = current_loss
            else:
                self.ema_losses[key] = (self.smoothing_factor * current_loss +
                                        (1 - self.smoothing_factor) * self.ema_losses[key])

        # 定期更新权重
        if self.update_counter % self.update_interval == 0:
            # 计算平均损失
            avg_losses = self.ema_losses.copy()

            # 计算总损失
            total_loss = sum(avg_losses.values())
            if total_loss < self.eps:
                return

            # 计算新的权重比例
            new_weights = {}
            for key in self.weights:
                # 损失占比 = 该项损失 / 总损失
                loss_ratio = avg_losses[key] / total_loss
                # 权重与损失占比成正比
                new_weights[key] = loss_ratio

            # 归一化权重
            total_weight = sum(new_weights.values())
            for key in new_weights:
                new_weights[key] /= total_weight

            # 应用新权重（平滑过渡）
            for key in self.weights:
                self.weights[key] = (0.5 * self.weights[key] +
                                     0.5 * new_weights[key])

            # 归一化最终权重
            total_weight = sum(self.weights.values())
            for key in self.weights:
                self.weights[key] /= total_weight

            # 打印权重更新信息
            print(f"更新权重: PDE={self.weights['pde']:.4f}, BC={self.weights['bc']:.4f}")

    def get_weighted_loss(self, losses):
        """计算加权损失"""
        total_loss = 0
        for key in losses:
            total_loss += self.weights[key] * losses[key]
        return total_loss


def main():
    """主函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

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

    # 生成训练点
    x_domain, y_domain = generate_domain_points(100, 100, device=device)

    # 初始化模型
    model = PINN().to(device)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 自适应损失权重
    if ADAPTIVE_LOSS:
        loss_weights = AdaptiveLossWeights({
            'pde': 1.0,
            'bc': 1.0
        })

    # 训练循环
    for epoch in range(EPOCHS):
        # 随机采样批次
        idx = torch.randint(0, len(x_domain), (BATCH_SIZE,))
        x_batch = x_domain[idx]
        y_batch = y_domain[idx]

        # 前向传播
        net_out = model(x_batch, y_batch)

        # 应用硬约束得到位移
        u_pred = physics.hard_constraint_displacement(x_batch, y_batch, net_out,
                                                      boundary_func_left, boundary_func_right,
                                                      boundary_func_top, boundary_func_bottom)

        # 计算物理残差
        res_x, res_y = physics.equilibrium_equations(u_pred, x_batch, y_batch)
        physics_loss = torch.mean(res_x ** 2 + res_y ** 2)

        # 计算边界条件损失
        bc_loss = torch.tensor(0.0, device=device)

        # 左侧边界 (x=0): 如果不是自由边界，计算损失
        if BOUNDARY_LEFT != 'free':
            left_mask = (x_batch <= 1e-5).squeeze()
            if left_mask.any():
                u_left = u_pred[left_mask]
                left_loss = torch.mean(u_left ** 2)
                bc_loss += left_loss

        # 右侧边界 (x=PLATE_WIDTH): 如果不是自由边界，计算损失
        if BOUNDARY_RIGHT != 'free':
            right_mask = (x_batch >= PLATE_WIDTH - 1e-5).squeeze()
            if right_mask.any():
                u_right = u_pred[right_mask]
                u_x_right = u_right[:, 0:1]
                u_y_right = u_right[:, 1:2]

                # 目标值
                u_x_bc, u_y_bc = boundary_func_right(x_batch[right_mask], y_batch[right_mask])

                # 边界条件损失
                right_loss_x = torch.mean((u_x_right - u_x_bc) ** 2)
                right_loss_y = torch.mean((u_y_right - u_y_bc) ** 2)
                bc_loss += right_loss_x + right_loss_y

        # 上边界 (y=PLATE_HEIGHT): 如果不是自由边界，计算损失
        if BOUNDARY_TOP != 'free':
            top_mask = (y_batch >= PLATE_HEIGHT - 1e-5).squeeze()
            if top_mask.any():
                u_top = u_pred[top_mask]
                u_x_top = u_top[:, 0:1]
                u_y_top = u_top[:, 1:2]

                # 目标值
                u_x_bc, u_y_bc = boundary_func_top(x_batch[top_mask], y_batch[top_mask])

                # 边界条件损失
                top_loss_x = torch.mean((u_x_top - u_x_bc) ** 2)
                top_loss_y = torch.mean((u_y_top - u_y_bc) ** 2)
                bc_loss += top_loss_x + top_loss_y

        # 下边界 (y=0): 如果不是自由边界，计算损失
        if BOUNDARY_BOTTOM != 'free':
            bottom_mask = (y_batch <= 1e-5).squeeze()
            if bottom_mask.any():
                u_bottom = u_pred[bottom_mask]
                u_x_bottom = u_bottom[:, 0:1]
                u_y_bottom = u_bottom[:, 1:2]

                # 目标值
                u_x_bc, u_y_bc = boundary_func_bottom(x_batch[bottom_mask], y_batch[bottom_mask])

                # 边界条件损失
                bottom_loss_x = torch.mean((u_x_bottom - u_x_bc) ** 2)
                bottom_loss_y = torch.mean((u_y_bottom - u_y_bc) ** 2)
                bc_loss += bottom_loss_x + bottom_loss_y

        # 总损失
        if ADAPTIVE_LOSS:
            losses = {
                'pde': physics_loss,
                'bc': bc_loss
            }
            total_loss = loss_weights.get_weighted_loss(losses)
            loss_weights.update(losses)
        else:
            total_loss = physics_loss + bc_loss

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 打印进度
        if (epoch + 1) % 100 == 0:
            if ADAPTIVE_LOSS:
                print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {total_loss.item():.6f}, "
                      f"Physics Loss: {physics_loss.item():.6f}, BC Loss: {bc_loss.item():.6f}, "
                      f"Weights: PDE={loss_weights.weights['pde']:.4f}, BC={loss_weights.weights['bc']:.4f}")
            else:
                print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {total_loss.item():.6f}, "
                      f"Physics Loss: {physics_loss.item():.6f}, BC Loss: {bc_loss.item():.6f}")

    # 训练完成后保存模型
    torch.save(model.state_dict(), 'pinn_model.pth')
    print("模型已保存为 'pinn_model.pth'")

    # 绘制结果
    model = model.cpu()
    plot_results(model, boundary_func_left, boundary_func_right, boundary_func_top, boundary_func_bottom)
    print("结果已保存为 'displacement_field.png', 'stress_strain_field.png' 和 'von_mises_stress.png'")

    # 运行边界可视化
    print("\n运行边界可视化...")
    from boundary_visualization import main as boundary_main
    boundary_main()


if __name__ == "__main__":
    main()