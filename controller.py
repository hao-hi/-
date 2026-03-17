"""
控制器模块
实现PD控制器和ADRC控制器接口
"""

import numpy as np
from core_utils import quat_error


def pd_torque(qd, q, w, Kp, Kd):
    """
    PD控制器：基于四元数误差的PD控制律
    
    参数:
        qd: 期望姿态四元数
        q: 当前姿态四元数
        w: 当前角速度 (rad/s)
        Kp: 比例增益
        Kd: 微分增益
    
    返回:
        u: 控制力矩 (N·m)
        q_e: 姿态误差四元数
    """
    q_e = quat_error(qd, q)
    # PD控制律：u = -Kp * q_e[1:4] - Kd * w
    # 注意：只使用四元数向量部分（虚部）
    u = -Kp * q_e[1:] - Kd * w
    return u, q_e
