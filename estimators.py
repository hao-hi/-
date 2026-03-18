"""
在线参数辨识模块
包含基于 ESO + RLS 的对角惯量在线辨识器
"""

from collections import deque

import numpy as np


def ensure_diag_inertia(J_like, min_inertia=1e-4):
    """
    将输入规范化为正的对角惯量向量 [Jxx, Jyy, Jzz]。
    """
    arr = np.asarray(J_like, dtype=float)
    if arr.ndim == 2:
        arr = np.diag(arr)
    arr = arr.reshape(-1)
    if arr.size != 3:
        raise ValueError("惯量输入必须可转换为长度 3 的对角向量")
    return np.maximum(arr[:3], float(min_inertia))


def build_inertia_regression_matrix(omega, wdot):
    """
    构造对角惯量辨识的线性回归矩阵 Phi。

    由欧拉方程
        J * wdot + w x (Jw) = u + d
    且 J = diag(Jxx, Jyy, Jzz) 可得
        y = Phi * theta
        theta = [Jxx, Jyy, Jzz]^T

    其中三轴逐行展开为：
        yx = ux + dx = [ wdot_x, -wy*wz,  wy*wz] * theta
        yy = uy + dy = [ wx*wz,  wdot_y, -wx*wz] * theta
        yz = uz + dz = [-wx*wy,  wx*wy,  wdot_z] * theta
    """
    wx, wy, wz = np.asarray(omega, dtype=float)
    wdx, wdy, wdz = np.asarray(wdot, dtype=float)
    return np.array(
        [
            [wdx, -wy * wz,  wy * wz],
            [wx * wz,  wdy, -wx * wz],
            [-wx * wy, wx * wy,  wdz],
        ],
        dtype=float,
    )


def diagonal_inertia_wdot_jacobian(J_diag, omega, torque):
    """
    计算对角惯量参数对角加速度模型 f = wdot 的偏导数:
        G_J = ∂wdot / ∂J

    角速度模型采用
        wdot = J^{-1} (u - w x (Jw))
    且 J = diag(Jxx, Jyy, Jzz)。
    """
    Jx, Jy, Jz = ensure_diag_inertia(J_diag)
    wx, wy, wz = np.asarray(omega, dtype=float)
    ux, uy, uz = np.asarray(torque, dtype=float)

    num_x = ux - (Jz - Jy) * wy * wz
    num_y = uy - (Jx - Jz) * wx * wz
    num_z = uz - (Jy - Jx) * wx * wy

    wdot_x = num_x / Jx
    wdot_y = num_y / Jy
    wdot_z = num_z / Jz

    return np.array(
        [
            [-wdot_x / Jx,  wy * wz / Jx, -wy * wz / Jx],
            [-wx * wz / Jy, -wdot_y / Jy, wx * wz / Jy],
            [wx * wy / Jz, -wx * wy / Jz, -wdot_z / Jz],
        ],
        dtype=float,
    )


class InertiaRLS:
    """
    基于 ESO + RLS 的对角惯量在线辨识器。

    说明:
        1) 量测 y 取为 u + d_hat，其中 d_hat 来自 ADRC 的 ESO 等效扰动力矩估计。
        2) 回归矩阵 Phi 由 build_inertia_regression_matrix(omega, wdot) 给出。
        3) 参数 theta = [Jxx, Jyy, Jzz]^T。
    """

    def __init__(
        self,
        J0,
        lambda_factor=0.98,
        P0=None,
        min_inertia=1e-4,
        max_inertia=5.0,
        min_accel_excitation=1e-3,
        min_torque_excitation=1e-3,
        max_update_step=5e-3,
        min_regressor_norm=1e-3,
        innovation_clip=0.08,
        regularization=1e-8,
        covariance_floor=1e-6,
        covariance_ceiling=50.0,
        excitation_alpha=0.10,
        innovation_deadzone=0.0,
        min_update_norm=0.0,
        window_size=8,
        theta_smoothing=0.60,
        filter_alpha=0.15,
        axis_weight_floor=0.20,
        axis_weight_power=0.50,
    ):
        self.theta = ensure_diag_inertia(J0, min_inertia=min_inertia)
        self.lambda_factor = float(np.clip(lambda_factor, 0.90, 1.0))
        self.P = (
            np.eye(3, dtype=float) * 1.0
            if P0 is None
            else np.asarray(P0, dtype=float).copy()
        )
        self.min_inertia = float(min_inertia)
        self.max_inertia = float(max_inertia)
        self.min_accel_excitation = float(min_accel_excitation)
        self.min_torque_excitation = float(min_torque_excitation)
        self.max_update_step = float(max_update_step)
        self.min_regressor_norm = float(min_regressor_norm)
        self.innovation_clip = float(innovation_clip)
        self.regularization = float(max(0.0, regularization))
        self.covariance_floor = float(max(1e-12, covariance_floor))
        self.covariance_ceiling = float(max(self.covariance_floor, covariance_ceiling))
        self.excitation_alpha = float(np.clip(excitation_alpha, 1e-3, 1.0))
        self.innovation_deadzone = float(max(0.0, innovation_deadzone))
        self.min_update_norm = float(max(0.0, min_update_norm))
        self.window_size = max(1, int(window_size))
        self.theta_smoothing = float(np.clip(theta_smoothing, 0.05, 1.0))
        self.Phi_f = np.zeros((3, 3), dtype=float)
        self.y_f = np.zeros(3, dtype=float)
        self.filter_alpha = float(np.clip(filter_alpha, 1e-3, 1.0))
        self.axis_weight_floor = float(np.clip(axis_weight_floor, 0.0, 1.0))
        self.axis_weight_power = float(np.clip(axis_weight_power, 0.05, 1.5))
        self.accel_level = 0.0
        self.torque_level = 0.0
        self.regressor_level = 0.0
        self.last_regressor_min_sv = 0.0
        self.last_axis_regressor_norms = np.zeros(3, dtype=float)
        self.axis_regressor_level = np.zeros(3, dtype=float)
        self._phi_window = deque(maxlen=self.window_size)
        self._y_window = deque(maxlen=self.window_size)

    def get_inertia_diag(self):
        """返回当前估计的对角惯量。"""
        return self.theta.copy()

    def get_last_regressor_min_sv(self):
        """返回最近一次回归矩阵的最小奇异值。"""
        return float(self.last_regressor_min_sv)

    def get_last_axis_regressor_norms(self):
        """返回最近一次各惯量参数列的等效回归强度。"""
        return self.last_axis_regressor_norms.copy()

    def update(self, omega, wdot, u, disturbance_torque=None):
        """
        执行一次 RLS 更新。

        参数:
            omega: 当前滤波角速度
            wdot:  当前离散差分角加速度
            u:     当前用于构造动力学方程的已施加控制力矩
            disturbance_torque: ESO 输出的等效扰动力矩估计 d_hat

        返回:
            theta:   更新后的对角惯量估计
            updated: 本次是否执行了 RLS 更新
        """
        omega = np.asarray(omega, dtype=float)
        wdot = np.asarray(wdot, dtype=float)
        u = np.asarray(u, dtype=float)
        if disturbance_torque is None:
            disturbance_torque = np.zeros(3, dtype=float)
        else:
            disturbance_torque = np.asarray(disturbance_torque, dtype=float)

        Phi = build_inertia_regression_matrix(omega, wdot)
        y = u + disturbance_torque
        alpha = float(np.clip(self.filter_alpha, 1e-6, 1.0))
        self.Phi_f = (1.0 - alpha) * self.Phi_f + alpha * Phi
        self.y_f = (1.0 - alpha) * self.y_f + alpha * y

        Phi_f = self.Phi_f.copy()
        y_f = self.y_f.copy()
        self._phi_window.append(Phi_f.copy())
        self._y_window.append(y_f.copy())

        Phi_stack = np.vstack(self._phi_window)
        y_stack = np.concatenate(self._y_window)
        window_count = max(1, len(self._phi_window))

        singular_values = np.linalg.svd(Phi_stack, compute_uv=False)
        self.last_regressor_min_sv = float(np.min(singular_values)) if singular_values.size else 0.0
        accel_norm = np.linalg.norm(wdot)
        torque_norm = np.linalg.norm(y_f)
        current_excitation = (np.linalg.norm(Phi_stack, ord='fro') ** 2) / float(window_count)
        regressor_norm = np.sqrt(current_excitation)
        axis_column_energy = np.sum(Phi_stack ** 2, axis=0) / float(window_count)
        axis_regressor_norms = np.sqrt(np.maximum(axis_column_energy, 0.0))
        self.last_axis_regressor_norms = axis_regressor_norms.copy()
        excitation_alpha = self.excitation_alpha
        self.accel_level = (1.0 - excitation_alpha) * self.accel_level + excitation_alpha * accel_norm
        self.torque_level = (1.0 - excitation_alpha) * self.torque_level + excitation_alpha * torque_norm
        self.regressor_level = (1.0 - excitation_alpha) * self.regressor_level + excitation_alpha * regressor_norm
        self.axis_regressor_level = (
            (1.0 - excitation_alpha) * self.axis_regressor_level
            + excitation_alpha * axis_regressor_norms
        )

        if (
            self.accel_level < self.min_accel_excitation
            or self.torque_level < self.min_torque_excitation
            or self.regressor_level < self.min_regressor_norm
        ):
            # 激励不足时冻结 theta 与 P，防止协方差风化导致参数漂移。
            return self.theta.copy(), False

        innovation = y_stack - Phi_stack @ self.theta
        innovation_norm = np.linalg.norm(innovation)
        innovation_deadzone = self.innovation_deadzone * np.sqrt(window_count)
        if (
            innovation_norm < innovation_deadzone
            or current_excitation < self.min_regressor_norm
        ):
            # 严格稳态死区：冻结 theta 与 P，切断稳态噪声驱动的参数漂移。
            return self.theta.copy(), False

        lambda_v = 1.0 - (1.0 - self.lambda_factor) * (1.0 - np.exp(-current_excitation / 0.01))
        lambda_v = float(np.clip(lambda_v, self.lambda_factor, 1.0))
        innovation_clip = self.innovation_clip * np.sqrt(window_count)
        if innovation_norm > innovation_clip:
            innovation = innovation * (innovation_clip / (innovation_norm + 1e-12))

        I3 = np.eye(3, dtype=float)
        gram = Phi_stack.T @ Phi_stack
        rhs = Phi_stack.T @ innovation
        prior_precision = np.linalg.inv(self.P)
        posterior_precision = (
            lambda_v * prior_precision
            + gram
            + self.regularization * I3
        )
        delta_theta = np.linalg.solve(posterior_precision, rhs)
        if np.linalg.norm(delta_theta) < self.min_update_norm:
            return self.theta.copy(), False
        max_axis_norm = float(np.max(axis_regressor_norms))
        if max_axis_norm <= 1e-12:
            return self.theta.copy(), False
        axis_strength = axis_regressor_norms / max_axis_norm
        axis_weights = np.maximum(
            self.axis_weight_floor,
            axis_strength ** self.axis_weight_power,
        )
        delta_theta = delta_theta * axis_weights
        delta_theta = np.clip(
            delta_theta,
            -self.max_update_step * axis_weights,
            self.max_update_step * axis_weights,
        )
        theta_candidate = self.theta + delta_theta
        self.theta = (1.0 - self.theta_smoothing) * self.theta + self.theta_smoothing * theta_candidate
        self.theta = np.clip(self.theta, self.min_inertia, self.max_inertia)

        self.P = np.linalg.inv(posterior_precision)
        self.P = 0.5 * (self.P + self.P.T)
        eigvals, eigvecs = np.linalg.eigh(self.P)
        eigvals = np.clip(eigvals, self.covariance_floor, self.covariance_ceiling)
        self.P = eigvecs @ np.diag(eigvals) @ eigvecs.T

        return self.theta.copy(), True
