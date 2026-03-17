"""
卫星动力学模块
实现刚体卫星的动力学模型
使用RK4积分提高精度
"""

import numpy as np
from config import CONFIG
from core_utils import quat_norm, omega_mat


class Spacecraft:
    """卫星动力学模型"""
    
    def __init__(self, J=None, umax=None):
        """
        初始化卫星参数
        
        参数:
            J: 转动惯量矩阵 (3x3)，默认使用config
            umax: 最大控制力矩幅值 (N·m)，默认使用config
        """
        if J is None:
            self.J = np.asarray(CONFIG['spacecraft']['inertia'], dtype=float)
        else:
            self.J = np.asarray(J, dtype=float)
        self._refresh_inertia_cache()
        if umax is None:
            self.umax = CONFIG['spacecraft']['u_max']
        else:
            self.umax = umax

    def _refresh_inertia_cache(self):
        """
        缓存惯量矩阵的常用形式，便于在积分热路径中走快速分支。
        """
        self.J_diag = np.diag(self.J).astype(float, copy=True)
        self._is_diagonal = bool(np.allclose(self.J, np.diag(self.J_diag)))
        if self._is_diagonal:
            self.J_inv_diag = 1.0 / np.maximum(self.J_diag, 1e-12)
            self.J_inv = np.diag(self.J_inv_diag)
        else:
            self.J_inv_diag = None
            self.J_inv = np.linalg.inv(self.J)

    def update_inertia(self, new_J):
        """
        在线更新转动惯量矩阵及其逆矩阵。

        参数:
            new_J: 新惯量，可为 3x3 矩阵或长度 3 的对角线元素
        """
        J_arr = np.asarray(new_J, dtype=float)
        if J_arr.ndim == 1:
            if J_arr.size != 3:
                raise ValueError("new_J 作为向量输入时必须包含 3 个对角惯量元素")
            J_mat = np.diag(np.maximum(J_arr, 1e-6))
        elif J_arr.shape == (3, 3):
            J_mat = J_arr.copy()
            diag = np.maximum(np.diag(J_mat), 1e-6)
            np.fill_diagonal(J_mat, diag)
        else:
            raise ValueError("new_J 必须是长度 3 的向量或 3x3 矩阵")

        self.J = J_mat
        self._refresh_inertia_cache()
    
    def step(self, q, w, u, dt, dist=None):
        """
        单步动力学积分（使用RK4方法）
        
        参数:
            q: 当前姿态四元数
            w: 当前角速度 (rad/s)
            u: 控制力矩 (N·m)
            dt: 时间步长 (s)
            dist: 外部干扰力矩 (N·m)
        
        返回:
            q_new: 新姿态四元数
            w_new: 新角速度
            u_sat: 饱和后的控制力矩
        """
        if dist is None:
            dist = np.zeros(3, dtype=float)

        # 力矩饱和
        u_sat = np.clip(u, -self.umax, self.umax)
        
        # RK4积分
        q_new, w_new = self._rk4_step(q, w, u_sat, dt, dist)
        
        return q_new, w_new, u_sat
    
    def _rk4_step(self, q, w, u, dt, dist):
        """
        RK4积分步
        
        参数:
            q, w: 当前状态
            u: 控制力矩
            dt: 时间步长
            dist: 扰动
        
        返回:
            q_new, w_new: 新状态
        """
        # k1
        k1_w = self._omega_derivative(w, u, dist)
        k1_q = 0.5 * omega_mat(w) @ q
        
        # k2
        w_temp = w + 0.5 * k1_w * dt
        q_temp = quat_norm(q + 0.5 * k1_q * dt)
        k2_w = self._omega_derivative(w_temp, u, dist)
        k2_q = 0.5 * omega_mat(w_temp) @ q_temp
        
        # k3
        w_temp = w + 0.5 * k2_w * dt
        q_temp = quat_norm(q + 0.5 * k2_q * dt)
        k3_w = self._omega_derivative(w_temp, u, dist)
        k3_q = 0.5 * omega_mat(w_temp) @ q_temp
        
        # k4
        w_temp = w + k3_w * dt
        q_temp = quat_norm(q + k3_q * dt)
        k4_w = self._omega_derivative(w_temp, u, dist)
        k4_q = 0.5 * omega_mat(w_temp) @ q_temp
        
        # 组合
        w_new = w + (dt/6) * (k1_w + 2*k2_w + 2*k3_w + k4_w)
        q_new = quat_norm(q + (dt/6) * (k1_q + 2*k2_q + 2*k3_q + k4_q))
        
        return q_new, w_new
    
    def _omega_derivative(self, w, u, dist):
        """
        计算角速度导数（欧拉方程）
        
        参数:
            w: 角速度
            u: 控制力矩
            dist: 扰动
        
        返回:
            wdot: 角速度导数
        """
        if self._is_diagonal:
            wx, wy, wz = np.asarray(w, dtype=float)
            Jx, Jy, Jz = self.J_diag
            torque = np.asarray(u, dtype=float) + np.asarray(dist, dtype=float)
            cross_x = (Jz - Jy) * wy * wz
            cross_y = (Jx - Jz) * wx * wz
            cross_z = (Jy - Jx) * wx * wy
            return np.array(
                [
                    (torque[0] - cross_x) * self.J_inv_diag[0],
                    (torque[1] - cross_y) * self.J_inv_diag[1],
                    (torque[2] - cross_z) * self.J_inv_diag[2],
                ],
                dtype=float,
            )

        Jw = self.J @ w
        wdot = self.J_inv @ (u - np.cross(w, Jw) + dist)
        return wdot
