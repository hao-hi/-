"""
在线参数辨识模块
包含基于 ESO + RLS 的对角惯量在线辨识器
"""

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

    def get_inertia_diag(self):
        """返回当前估计的对角惯量。"""
        return self.theta.copy()

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

        if (
            np.linalg.norm(wdot) < self.min_accel_excitation
            or np.linalg.norm(u) < self.min_torque_excitation
        ):
            # 激励不足时冻结 theta 与 P，防止协方差风化导致参数漂移。
            return self.theta.copy(), False

        Phi = build_inertia_regression_matrix(omega, wdot)
        y = u + disturbance_torque

        S = self.lambda_factor * np.eye(3, dtype=float) + Phi @ self.P @ Phi.T
        PHt = self.P @ Phi.T
        K = np.linalg.solve(S.T, PHt.T).T
        innovation = y - Phi @ self.theta

        delta_theta = K @ innovation
        delta_theta = np.clip(delta_theta, -self.max_update_step, self.max_update_step)
        self.theta = self.theta + delta_theta
        self.theta = np.clip(self.theta, self.min_inertia, self.max_inertia)

        I3 = np.eye(3, dtype=float)
        self.P = (I3 - K @ Phi) @ self.P / self.lambda_factor
        self.P = 0.5 * (self.P + self.P.T)

        return self.theta.copy(), True
