# config.py - 配置参数

# 几何参数
PLATE_WIDTH = 1.0   # 平板宽度 (m)
PLATE_HEIGHT = 1.0  # 平板高度 (m)

# 材料参数 (铜)
YOUNG_MODULUS = 110e9  # 杨氏模量 (Pa)
POISSON_RATIO = 0.34   # 泊松比
DENSITY = 8960         # 密度 (kg/m³)

# 神经网络参数
HIDDEN_LAYERS = 5      # 隐藏层数量
NEURONS_PER_LAYER = 50 # 每层神经元数量
ACTIVATION = 'tanh'    # 激活函数

# 训练参数
EPOCHS = 2000         # 训练轮数
BATCH_SIZE = 1024      # 批次大小
LEARNING_RATE = 0.001  # 学习率
ADAPTIVE_LOSS = True   # 使用自适应损失权重

# 边界条件类型配置
# 可选值: 'function', 'interpolation', 'free', 'fixed'
BOUNDARY_LEFT = 'fixed'        # 左侧边界类型
BOUNDARY_RIGHT = 'interpolation'    # 右侧边界类型
BOUNDARY_TOP = 'free'          # 上侧边界类型
BOUNDARY_BOTTOM = 'free'      # 下侧边界类型

# 插值边界配置
INTERPOLATION_POINTS = 10     # 插值点数