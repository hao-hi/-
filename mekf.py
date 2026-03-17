"""
MEKF (Multiplicative Extended Kalman Filter) 模块
用于姿态估计，支持陀螺仪偏置估计
"""

import numpy as np
from core_utils import (
    quat_conjugate,
    quat_from_omega,
    quat_mul,
    quat_multiply,
    quat_normalize,
    quat_to_small_angle,
    small_angle_quat,
)
from estimators import ensure_diag_inertia, diagonal_inertia_wdot_jacobian


def small_angle_from_quat(dq):
    """从误差四元数提取小角度向量"""
    norm = np.linalg.norm(dq)
    if norm < 1e-15:
        return np.zeros(3, dtype=float)
    dq = dq / norm
    if dq[0] < 0.0:
        dq = -dq
    return 2.0 * dq[1:]


def quat_measurement_delta(q_est, q_meas):
    """
    计算量测四元数相对当前估计的姿态残差。
    """
    w, x, y, z = q_est
    qi = np.array([w, -x, -y, -z], dtype=float)
    dq = np.array(
        [
            q_meas[0] * qi[0] - np.dot(q_meas[1:], qi[1:]),
            q_meas[0] * qi[1]
            + qi[0] * q_meas[1]
            + q_meas[2] * qi[3]
            - q_meas[3] * qi[2],
            q_meas[0] * qi[2]
            - qi[3] * q_meas[1]
            + qi[0] * q_meas[2]
            + q_meas[1] * qi[3],
            q_meas[0] * qi[3]
            + qi[2] * q_meas[1]
            - qi[1] * q_meas[2]
            + qi[0] * q_meas[3],
        ],
        dtype=float,
    )
    return small_angle_from_quat(dq)


class MEKFBiasOnly:
    """
    带偏置估计的MEKF
    误差状态：x = [δθ, δb] ∈ R⁶
        δθ: 姿态小角度误差（3）
        δb: 陀螺仪偏置误差（3）
    """
    
    def __init__(
        self,
        q0=None,
        b0=None,
        P0=None,
        Q=None,
        R=None,
    ):
        """
        初始化MEKF（姿态 + 三轴陀螺偏置）
        
        参数:
            q0: 初始姿态四元数
            b0: 初始陀螺仪偏置 (3,)
            P0: 初始误差协方差矩阵 (6x6)
            Q: 过程噪声协方差 (6x6)，前3维为角度误差，后3维为偏置随机游走
            R: 测量噪声协方差 (3x3)，姿态测量的小角度噪声
        """
        # 名义状态
        self.q = (
            np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            if q0 is None
            else q0 / np.linalg.norm(q0)
        )
        self.b = np.zeros(3) if b0 is None else np.array(b0, dtype=float).copy()
        self._I3 = np.eye(3)
        self._I6 = np.eye(6)
        self._Z3 = np.zeros((3, 3))
        self._H = np.hstack([self._I3, self._Z3])
        self.covariance_floor = 1e-12
        self.covariance_ceiling = 1e12
        self.exact_covariance_projection = False

        # 误差协方差 P ∈ R^{6x6}
        if P0 is None:
            # 初始对姿态/偏置都有较小不确定性
            P_att = 1e-4 * self._I3
            P_bias = 1e-6 * self._I3
            self.P = np.block(
                [
                    [P_att, self._Z3],
                    [self._Z3, P_bias],
                ]
            )
        else:
            self.P = np.asarray(P0, dtype=float).copy()

        # 过程噪声 Q ∈ R^{6x6}
        if Q is None:
            # 姿态过程噪声略大于偏置随机游走
            Q_att = 1e-7 * self._I3
            Q_bias = 1e-10 * self._I3
            self.Q = np.block(
                [
                    [Q_att, self._Z3],
                    [self._Z3, Q_bias],
                ]
            )
        else:
            self.Q = np.asarray(Q, dtype=float).copy()

        # 测量噪声 R ∈ R^{3x3}
        self.R = 1e-5 * self._I3 if R is None else np.asarray(R, dtype=float).copy()

    def _symmetrize_with_bounds(self, P, floor=None, ceiling=None):
        floor = self.covariance_floor if floor is None else float(max(1e-12, floor))
        ceiling = self.covariance_ceiling if ceiling is None else float(max(floor, ceiling))
        P = 0.5 * (P + P.T)
        if not self.exact_covariance_projection:
            diag_idx = np.diag_indices_from(P)
            P[diag_idx] = np.clip(P[diag_idx], floor, ceiling)
            return P
        eigvals, eigvecs = np.linalg.eigh(P)
        eigvals = np.clip(eigvals, floor, ceiling)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    def predict(self, omega_meas, dt):
        """
        预测步骤：使用陀螺仪测量值进行姿态传播
        
        参数:
            omega_meas: 陀螺仪测量的角速度（含偏置）
            dt: 时间步长
        """
        # 使用去偏置后的角速度进行姿态传播
        dq = quat_from_omega(omega_meas - self.b, dt)
        self.q = quat_mul(self.q, dq)
        self.q = self.q / np.linalg.norm(self.q)
        
        # 误差状态线性化模型（简化）：
        #   δθ_k+1 ≈ δθ_k - δb_k * dt
        #   δb_k+1 ≈ δb_k
        # 离散状态转移矩阵 Φ ∈ R^{6x6}
        Phi = np.block(
            [
                [self._I3, -dt * self._I3],
                [self._Z3, self._I3],
            ]
        )
        
        # 协方差传播
        self.P = self._symmetrize_with_bounds(Phi @ self.P @ Phi.T + self.Q * dt)
    
    def update(self, q_meas):
        """
        更新步骤：使用姿态测量值更新估计
        
        参数:
            q_meas: 测量得到的姿态四元数
        
        返回:
            delta: 姿态测量残差（小角度向量）
            dx:    误差状态更新量（[δθ, δb]）
        """
        delta = quat_measurement_delta(self.q, q_meas)
        
        # 量测方程：delta ≈ H * x + v,  x = [δθ; δb]
        H = self._H

        S = H @ self.P @ H.T + self.R          # 3x3
        PHt = self.P @ H.T
        K = np.linalg.solve(S.T, PHt.T).T      # 6x3
        
        # 误差状态更新
        dx = K @ delta                         # 6x1
        dtheta = dx[0:3]
        db = dx[3:6]
        
        # 更新偏置估计
        self.b += db
        
        KH = K @ H
        joseph = (self._I6 - KH) @ self.P @ (self._I6 - KH).T + K @ self.R @ K.T
        self.P = self._symmetrize_with_bounds(joseph)
        
        # 更新姿态估计：q_new = δq(δθ) ⊗ q_old
        corr = np.array([1.0, *(0.5 * dtheta)])
        self.q = quat_mul(corr, self.q)
        self.q = self.q / np.linalg.norm(self.q)
        
        return delta, dx


class MEKF_Augmented(MEKFBiasOnly):
    """
    增广 MEKF：在姿态误差与陀螺偏置之外，额外在线估计对角惯量偏差。

    误差状态:
        X = [δθ, δb, ΔJ]^T ∈ R^9

    建模说明:
        1) ΔJ 采用随机游走模型：dot(J) = w_J
        2) 采用当前名义惯量 J_hat 与上一拍施加力矩 u_applied，构造
           G_J = ∂wdot/∂J
        3) 由于本工程未显式把 δω 纳入误差状态，离散化时用
           δθ_{k+1} ≈ δθ_k - δb_k*dt + 0.5*dt^2*G_J*ΔJ
           将惯量误差对姿态误差的影响折算进 Φ 的右上块
    """

    def __init__(
        self,
        q0=None,
        b0=None,
        J0=None,
        P0=None,
        Q=None,
        R=None,
        inertia_bounds=(1e-4, 5.0),
        dynamics_measurement_noise=0.15,
        min_dynamics_excitation=0.05,
        min_regressor_norm=2e-3,
        dynamics_innovation_clip=0.6,
        max_inertia_step=8e-4,
        inertia_update_gain=0.65,
        covariance_floor=1e-9,
        covariance_ceiling=1.0,
    ):
        self.inertia_bounds = (float(inertia_bounds[0]), float(inertia_bounds[1]))
        self.J_diag = ensure_diag_inertia(
            np.array([0.08, 0.07, 0.05], dtype=float) if J0 is None else J0,
            min_inertia=self.inertia_bounds[0],
        )
        self.last_u_applied = np.zeros(3, dtype=float)
        self.last_wdot_model = np.zeros(3, dtype=float)
        self.last_omega_corr = np.zeros(3, dtype=float)
        self.last_G_J = np.zeros((3, 3), dtype=float)
        self.last_dynamics_update_active = False
        self.dynamics_measurement_noise = float(max(1e-6, dynamics_measurement_noise))
        self.min_dynamics_excitation = float(max(0.0, min_dynamics_excitation))
        self.min_regressor_norm = float(max(0.0, min_regressor_norm))
        self.dynamics_innovation_clip = float(max(1e-6, dynamics_innovation_clip))
        self.max_inertia_step = float(max(1e-6, max_inertia_step))
        self.inertia_update_gain = float(np.clip(inertia_update_gain, 0.05, 1.0))
        self.covariance_floor = float(max(1e-12, covariance_floor))
        self.covariance_ceiling = float(max(self.covariance_floor, covariance_ceiling))
        self.exact_covariance_projection = True

        super().__init__(q0=q0, b0=b0, P0=None, Q=None, R=R)

        if P0 is None:
            P_att = 1e-4 * self._I3
            P_bias = 1e-6 * self._I3
            P_J = 5e-4 * self._I3
            self.P = np.block(
                [
                    [P_att, self._Z3, self._Z3],
                    [self._Z3, P_bias, self._Z3],
                    [self._Z3, self._Z3, P_J],
                ]
            )
        else:
            self.P = np.asarray(P0, dtype=float).copy()

        if Q is None:
            Q_att = 1e-7 * self._I3
            Q_bias = 1e-10 * self._I3
            Q_J = 5e-10 * self._I3
            self.Q = np.block(
                [
                    [Q_att, self._Z3, self._Z3],
                    [self._Z3, Q_bias, self._Z3],
                    [self._Z3, self._Z3, Q_J],
                ]
            )
        else:
            self.Q = np.asarray(Q, dtype=float).copy()

    def get_inertia_diag(self):
        """返回当前估计的对角惯量。"""
        return self.J_diag.copy()

    def get_inertia_matrix(self):
        """返回当前估计的惯量矩阵。"""
        return np.diag(self.J_diag)

    def _compute_wdot_model(self, omega_corr, torque):
        omega_corr = np.asarray(omega_corr, dtype=float)
        torque = np.asarray(torque, dtype=float)
        return (
            torque
            - np.cross(omega_corr, self.J_diag * omega_corr)
        ) / np.maximum(self.J_diag, 1e-9)

    def predict(self, omega_meas, dt, u_applied=None):
        """
        增广预测步骤。

        参数:
            omega_meas: 陀螺仪测量角速度（含偏置）
            dt: 时间步长
            u_applied: 上一拍实际施加力矩，用于构造 ∂wdot/∂J
        """
        omega_corr = np.asarray(omega_meas, dtype=float) - self.b
        self.last_omega_corr = omega_corr.copy()
        dq = quat_from_omega(omega_corr, dt)
        self.q = quat_mul(self.q, dq)
        self.q = self.q / np.linalg.norm(self.q)

        u_use = self.last_u_applied if u_applied is None else np.asarray(u_applied, dtype=float)
        self.last_u_applied = u_use.copy()

        G_J = diagonal_inertia_wdot_jacobian(self.J_diag, omega_corr, u_use)
        self.last_G_J = G_J.copy()
        self.last_wdot_model = self._compute_wdot_model(omega_corr, u_use)

        I3 = np.eye(3)
        zeros3 = np.zeros((3, 3))
        Phi = np.block(
            [
                [I3, -dt * I3, 0.5 * (dt ** 2) * G_J],
                [zeros3, I3, zeros3],
                [zeros3, zeros3, I3],
            ]
        )

        self.P = self._symmetrize_with_bounds(
            Phi @ self.P @ Phi.T + self.Q * dt,
            floor=self.covariance_floor,
            ceiling=self.covariance_ceiling,
        )

    def update(self, q_meas):
        """
        增广更新步骤：姿态量测仍只直接观测 δθ，
        通过预测阶段形成的交叉协方差间接修正 ΔJ。
        """
        delta = quat_measurement_delta(self.q, q_meas)
        H = np.hstack([np.eye(3), np.zeros((3, 6))])

        S = H @ self.P @ H.T + self.R
        PHt = self.P @ H.T
        K = np.linalg.solve(S.T, PHt.T).T

        dx = K @ delta
        dtheta = dx[0:3]
        db = dx[3:6]
        dJ = self.inertia_update_gain * dx[6:9]

        self.b += db
        self.J_diag = np.clip(
            self.J_diag + dJ,
            self.inertia_bounds[0],
            self.inertia_bounds[1],
        )

        I9 = np.eye(9)
        joseph = (I9 - K @ H) @ self.P @ (I9 - K @ H).T + K @ self.R @ K.T
        self.P = self._symmetrize_with_bounds(
            joseph,
            floor=self.covariance_floor,
            ceiling=self.covariance_ceiling,
        )

        corr = np.array([1.0, *(0.5 * dtheta)])
        self.q = quat_mul(corr, self.q)
        self.q = self.q / np.linalg.norm(self.q)

        self.last_dynamics_update_active = False
        return delta, dx

    def update_dynamics(self, wdot_meas, disturbance_torque=None):
        """
        使用角加速度伪量测直接修正惯量参数。
        """
        wdot_meas = np.asarray(wdot_meas, dtype=float)
        disturbance = (
            np.zeros(3, dtype=float)
            if disturbance_torque is None
            else np.asarray(disturbance_torque, dtype=float)
        )
        effective_torque = self.last_u_applied + disturbance
        G_J = diagonal_inertia_wdot_jacobian(self.J_diag, self.last_omega_corr, effective_torque)

        excitation = max(
            np.linalg.norm(wdot_meas),
            np.linalg.norm(effective_torque),
        )
        if (
            excitation < self.min_dynamics_excitation
            or np.linalg.norm(G_J, ord='fro') < self.min_regressor_norm
        ):
            self.last_dynamics_update_active = False
            return False

        wdot_model = self._compute_wdot_model(self.last_omega_corr, effective_torque)
        innovation = wdot_meas - wdot_model
        innovation_norm = np.linalg.norm(innovation)
        if innovation_norm > self.dynamics_innovation_clip:
            innovation = innovation * (self.dynamics_innovation_clip / (innovation_norm + 1e-12))

        H_dyn = np.hstack([np.zeros((3, 6)), G_J])
        R_dyn = (self.dynamics_measurement_noise ** 2) * np.eye(3)
        S = H_dyn @ self.P @ H_dyn.T + R_dyn
        PHt = self.P @ H_dyn.T
        K = np.linalg.solve(S.T, PHt.T).T

        dx = K @ innovation
        dx[6:9] = np.clip(dx[6:9], -self.max_inertia_step, self.max_inertia_step)
        dx[6:9] *= self.inertia_update_gain

        self.b += dx[3:6]
        self.J_diag = np.clip(
            self.J_diag + dx[6:9],
            self.inertia_bounds[0],
            self.inertia_bounds[1],
        )

        I9 = np.eye(9)
        joseph = (I9 - K @ H_dyn) @ self.P @ (I9 - K @ H_dyn).T + K @ R_dyn @ K.T
        self.P = self._symmetrize_with_bounds(
            joseph,
            floor=self.covariance_floor,
            ceiling=self.covariance_ceiling,
        )
        self.last_wdot_model = wdot_model
        self.last_G_J = G_J
        self.last_dynamics_update_active = True
        return True


# 兼容demo2.py的简单MEKF类
class MEKF:
    """
    简化版MEKF（兼容demo2.py）
    只估计姿态，不估计偏置
    """
    
    def __init__(self, Q_err, R_meas):
        """
        初始化MEKF
        
        参数:
            Q_err: 误差过程噪声协方差
            R_meas: 测量噪声协方差
        """
        self.q_est = np.array([1.0, 0.0, 0.0, 0.0])
        self.P = np.eye(3)
        self.Q = Q_err
        self.R = R_meas
    
    def predict(self, delta_q_meas):
        """
        预测步骤：利用旋转增量更新估计
        
        参数:
            delta_q_meas: 旋转增量四元数
        """
        self.q_est = quat_normalize(quat_multiply(self.q_est, delta_q_meas))
        Phi = np.eye(3)
        self.P = Phi @ self.P @ Phi.T + self.Q
    
    def update(self, q_meas):
        """
        更新步骤：使用测量姿态更新估计
        
        参数:
            q_meas: 测量得到的姿态四元数
        """
        # 计算误差四元数
        q_est_inv = quat_conjugate(self.q_est)
        q_err = quat_multiply(q_meas, q_est_inv)
        q_err = quat_normalize(q_err)
        
        # 转换为小角度向量
        delta_theta_meas = quat_to_small_angle(q_err)
        
        # 卡尔曼滤波更新
        H = np.eye(3)
        S = H @ self.P @ H.T + self.R
        PHt = self.P @ H.T
        K = np.linalg.solve(S.T, PHt.T).T
        
        # 更新协方差
        self.P = (np.eye(3) - K @ H) @ self.P
        
        # 更新姿态估计
        delta_theta_update = K @ delta_theta_meas
        delta_q_update = small_angle_quat(delta_theta_update)
        self.q_est = quat_normalize(quat_multiply(delta_q_update, self.q_est))
    
    def get_estimate(self):
        """获取当前姿态估计"""
        return self.q_est
    
    def get_covariance(self):
        """获取当前协方差矩阵"""
        return self.P
