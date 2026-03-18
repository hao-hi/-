import numpy as np

# 卫星姿态控制仿真系统配置文件
# 集中管理所有参数，避免硬编码

CONFIG = {
    'spacecraft': {
        'inertia': np.diag([0.80, 0.70, 0.50]),  # kg·m²，转动惯量矩阵
        'u_max': 0.2,  # N·m，控制力矩饱和限制
        'disturbance': {
            'constant': np.array([0.001, 0.001, 0.001]),  # N·m，常值扰动
            'noise_std': 0.0001,  # N·m，扰动噪声标准差
        }
    },
    'simulation': {
        'T': 20.0,  # s，总仿真时间
        'dt': 0.03,  # s，时间步长
        'seed': 0,  # 随机种子
        'initial_conditions': {
            'q0': np.array([1.0, 0.0, 0.0, 0.0]),  # 初始姿态四元数（标量在前的单位四元数）
            'w0': np.array([0.1, 0.1, 0.1]),  # rad/s，初始角速度
        }
    },
    'sensors': {
        'star_tracker': {
            'fov': 20,  # deg，视场角
            'noise_std': 6e-4,  # rad，测量噪声标准差
            'n_stars': 1600,  # 恒星数量
        },
        'gyro': {
            'bias_std': 0.002,  # rad/s，偏置标准差
            'noise_std': 0.001,  # rad/s，噪声标准差
        }
    },
    'control': {
        'PD': {
            'Kp_range': (1, 8),  # 比例增益范围
            'Kd_range': (0.1, 2),  # 微分增益范围
        },
        'ADRC': {
            'omega_c': 2.5,  # 控制器带宽
            'omega_o': 8.0,  # 观测器带宽
            'b0': 1.0,  # 控制增益
        }
    },
    'optimization': {
        'weights': {
            'settle': 1.0,
            'overshoot': 1.0,
            'error': 1.0,
            'effort': 0.1,
            'saturation': 10.0,
        },
        'grid_points': 8,  # 网格搜索点数
        'max_iters': 100,  # 最大迭代次数
    }
}

# 便捷访问函数
def get_spacecraft_params():
    return CONFIG['spacecraft']

def get_sim_params():
    return CONFIG['simulation']

def get_sensor_params():
    return CONFIG['sensors']

def get_control_params():
    return CONFIG['control']

def get_opt_params():
    return CONFIG['optimization']
