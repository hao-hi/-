"""
核心工具函数模块
包含四元数运算、坐标转换、姿态积分等基础函数
所有模块共享，避免重复代码
"""

import numpy as np
try:
    from numba import jit as _numba_jit

    def jit(*args, **kwargs):
        kwargs.setdefault("cache", True)
        return _numba_jit(*args, **kwargs)
except Exception:
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@jit(nopython=True)
def quat_mul(q1, q2):
    """四元数乘法"""
    a = np.asarray(q1, dtype=np.float64)
    b = np.asarray(q2, dtype=np.float64)
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


@jit(nopython=True)
def quat_inv(q):
    """四元数逆"""
    qf = np.asarray(q, dtype=np.float64)
    w, x, y, z = qf
    return np.array([w, -x, -y, -z]) / (np.dot(qf, qf) + 1e-15)


@jit(nopython=True)
def quat_norm(q):
    """四元数归一化"""
    qf = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(qf)
    if norm < 1e-15:
        return qf.copy()
    return qf / norm


@jit(nopython=True)
def quat_from_axis_angle(axis, ang):
    """从轴角表示构造四元数"""
    a = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(a)
    if norm < 1e-15:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    a = a / norm
    s = np.sin(ang / 2.0)
    return np.array([np.cos(ang / 2.0), a[0]*s, a[1]*s, a[2]*s])


@jit(nopython=True)
def quat_to_R(q):
    """四元数转旋转矩阵"""
    q = quat_norm(q)
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ])


@jit(nopython=True)
def R_to_quat(R):
    """旋转矩阵转四元数"""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        d0 = R[0, 0]
        d1 = R[1, 1]
        d2 = R[2, 2]
        i = 0
        if d1 > d0 and d1 >= d2:
            i = 1
        elif d2 > d0 and d2 > d1:
            i = 2
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z])
    return quat_norm(q)


@jit(nopython=True)
def omega_mat(w):
    """角速度矩阵（用于四元数微分方程）"""
    wx, wy, wz = w
    return np.array([
        [0, -wx, -wy, -wz],
        [wx, 0, wz, -wy],
        [wy, -wz, 0, wx],
        [wz, wy, -wx, 0]
    ])

@jit(nopython=True)
def quat_from_omega(w, dt):
    """从角速度和时间步长构造四元数增量"""
    n = np.linalg.norm(w)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = w / n
    s = np.sin(n * dt / 2.0)
    dq = np.empty(4, dtype=np.float64)
    dq[0] = np.cos(n * dt / 2.0)
    dq[1] = axis[0] * s
    dq[2] = axis[1] * s
    dq[3] = axis[2] * s
    return dq


@jit(nopython=True)
def quat_error(qd, q):
    """计算姿态误差四元数（qd^{-1} ⊗ q）"""
    qi = quat_inv(qd)
    q_e = quat_mul(qi, q)
    if q_e[0] < 0:
        q_e = -q_e
    return q_e


def quat_angle_errors_deg(q_ref_seq, q_seq):
    """
    批量计算四元数序列之间的姿态角误差（单位：度）。
    """
    q_ref_arr = np.asarray(q_ref_seq, dtype=float)
    q_arr = np.asarray(q_seq, dtype=float)

    if q_ref_arr.ndim == 1:
        q_ref_arr = np.broadcast_to(q_ref_arr, q_arr.shape)
    elif q_ref_arr.shape != q_arr.shape:
        raise ValueError("q_ref_seq 与 q_seq 的形状必须一致，或 q_ref_seq 为单个四元数")

    q_ref_norm = q_ref_arr / np.maximum(np.linalg.norm(q_ref_arr, axis=1, keepdims=True), 1e-12)
    q_norm = q_arr / np.maximum(np.linalg.norm(q_arr, axis=1, keepdims=True), 1e-12)
    dots = np.abs(np.sum(q_ref_norm * q_norm, axis=1))
    dots = np.clip(dots, -1.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(dots))


def rand_unit(n, rng=None):
    """生成 n 个随机单位向量。"""
    randn = np.random.randn if rng is None else rng.randn
    v = randn(n, 3)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v
