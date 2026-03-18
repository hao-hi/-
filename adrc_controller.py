"""
线性自抗扰控制器（LADRC）模块
改进结构：TD + ESO + Feedback (Derivative on Measurement)
"""

import numpy as np
import warnings

from core_utils import quat_error, quat_from_omega, quat_mul, quat_norm


class ADRCController:
    """
    三轴并行 LADRC 控制器（TD-ESO-Feedback）。

    结构说明:
    1) TD: 平滑姿态指令 qd，输出 qd_smooth 与参考角速度 w_target
    2) ESO: 估计误差状态与总扰动 z3
    3) 反馈律: 使用陀螺仪测量角速度 w_gyro 做阻尼
       u = (Kp * e - Kd * w_gyro - z3) / b0
    """

    def __init__(
        self,
        b0=None,
        omega_c=None,
        omega_o=None,
        wc=None,
        beta1=None,
        beta2=None,
        beta3=None,
        kp=None,
        kd=None,
        td_omega=None,
        td_zeta=1.0,
        td_max_rate=None,
        td_max_acc=None,
        dt=None,
        **kwargs,
    ):
        """
        参数:
            b0: 控制增益，可为标量或长度3向量。默认按 J=diag([0.8,0.8,0.8]) 取 1/J。
            omega_c: 控制器带宽(rad/s)，用于 Kp=omega_c^2, Kd=2*omega_c。
            omega_o: ESO带宽(rad/s)，用于 beta1/2/3。
            wc: 兼容旧参数，等价于 omega_c。
            beta1,beta2,beta3: 若提供则优先反推/覆盖 omega_o。
            kp,kd: 可直接覆盖反馈增益。
            td_omega: TD带宽(rad/s)。兼容旧参数 w0。
            td_zeta: TD阻尼比，默认1.0（临界阻尼）。
            td_max_rate: TD输出角速度限幅(rad/s)，标量或3轴向量。
            td_max_acc: TD角加速度限幅(rad/s^2)，标量或3轴向量。
            dt: 采样时间，默认0.03s。
            kwargs: 兼容旧接口其余字段（alpha/delta/h等），将被忽略。
        """
        self.dt = float(dt) if dt is not None else 0.03

        # 兼容旧接口
        if omega_c is None:
            omega_c = float(wc) if wc is not None else 3.5
        if td_omega is None:
            td_omega = float(kwargs.get("w0", 2.2))

        # 若给了 beta 参数，按其反推 omega_o；否则采用默认比例
        if omega_o is None:
            if beta3 is not None:
                omega_o = np.cbrt(float(beta3))
            elif beta2 is not None:
                omega_o = np.sqrt(float(beta2) / 3.0)
            elif beta1 is not None:
                omega_o = float(beta1) / 3.0
            else:
                omega_o = 3.0 * float(omega_c)

        self.omega_c = float(omega_c)
        self.omega_o = float(omega_o)

        # 欧拉离散安全约束（经验）
        max_omega_o = 0.5 / max(self.dt, 1e-6)
        if self.omega_o > max_omega_o:
            warnings.warn(
                f"omega_o={self.omega_o:.4f} 对于 dt={self.dt}s 过高"
                f"（omega_o*dt={self.omega_o*self.dt:.4f} > 0.5），已自动降至 {max_omega_o:.4f}",
                RuntimeWarning,
                stacklevel=2,
            )
            self.omega_o = max_omega_o

        # Gao 带宽参数化
        self.beta1 = 3.0 * self.omega_o if beta1 is None else float(beta1)
        self.beta2 = 3.0 * (self.omega_o ** 2) if beta2 is None else float(beta2)
        self.beta3 = (self.omega_o ** 3) if beta3 is None else float(beta3)

        self.kp = float(self.omega_c ** 2 if kp is None else kp)
        self.kd = float(2.0 * self.omega_c if kd is None else kd)

        # b0 支持标量 / 三轴向量
        if b0 is None:
            self.b0 = np.full(3, 1.0 / 0.8, dtype=float)
        else:
            self.b0 = self._vec3(b0)
        self.b0 = np.maximum(self.b0, 1e-6)

        # TD 参数
        self.td_omega = float(td_omega)
        self.td_zeta = float(td_zeta)
        self.td_max_rate = None if td_max_rate is None else self._vec3(td_max_rate)
        self.td_max_acc = None if td_max_acc is None else self._vec3(td_max_acc)

        # ESO 状态: z1≈e, z2≈de/dt, z3≈总扰动
        self.z1 = np.zeros(3, dtype=float)
        self.z2 = np.zeros(3, dtype=float)
        self.z3 = np.zeros(3, dtype=float)

        # TD 状态
        self.qd_smooth = None
        self.w_target = np.zeros(3, dtype=float)

        # 内部记忆：上一时刻实际施加力矩（用于 ESO 反饱和）
        self.u_applied = np.zeros(3, dtype=float)

    @staticmethod
    def _vec3(x):
        arr = np.asarray(x, dtype=float).reshape(-1)
        if arr.size == 1:
            return np.full(3, arr[0], dtype=float)
        if arr.size >= 3:
            return arr[:3].astype(float)
        return np.pad(arr, (0, 3 - arr.size), mode="edge").astype(float)

    def _td_step(self, qd, dt):
        """
        二阶 TD（四元数参考调节器）：
            w_dot = td_omega^2 * r - 2*td_zeta*td_omega*w
        其中 r 为 qd_smooth 到 qd 的小角度旋转向量近似。
        """
        qd = quat_norm(np.asarray(qd, dtype=float))

        if self.qd_smooth is None:
            self.qd_smooth = qd.copy()
            self.w_target[:] = 0.0
            return self.qd_smooth, self.w_target

        if np.dot(self.qd_smooth, qd) < 0.0:
            qd = -qd

        q_ref = quat_error(self.qd_smooth, qd)
        r = 2.0 * q_ref[1:]

        w_dot = (self.td_omega ** 2) * r - 2.0 * self.td_zeta * self.td_omega * self.w_target

        if self.td_max_acc is not None:
            w_dot = np.clip(w_dot, -self.td_max_acc, self.td_max_acc)

        self.w_target = self.w_target + dt * w_dot

        if self.td_max_rate is not None:
            self.w_target = np.clip(self.w_target, -self.td_max_rate, self.td_max_rate)

        dq = quat_from_omega(self.w_target, dt)
        self.qd_smooth = quat_norm(quat_mul(self.qd_smooth, dq))

        if np.dot(self.qd_smooth, qd) < 0.0:
            self.qd_smooth = -self.qd_smooth

        return self.qd_smooth, self.w_target

    def eso_step(self, y, u_applied, dt):
        """
        线性 ESO：
            e_obs = z1 - y
            z1_dot = z2 - beta1*e_obs
            z2_dot = z3 - beta2*e_obs + b0*u_applied
            z3_dot = -beta3*e_obs

        注意：此处 u_applied 必须是“实际施加到执行机构”的力矩（含饱和后），
        避免观测器在饱和时发生等效积分风up。
        """
        y = np.asarray(y, dtype=float)
        u_applied = np.asarray(u_applied, dtype=float)

        e_obs = self.z1 - y
        z1_dot = self.z2 - self.beta1 * e_obs
        z2_dot = self.z3 - self.beta2 * e_obs + self.b0 * u_applied
        z3_dot = -self.beta3 * e_obs

        self.z1 = self.z1 + dt * z1_dot
        self.z2 = self.z2 + dt * z2_dot
        self.z3 = self.z3 + dt * z3_dot

    def compute_control(self, qd, q, w, dt, u_prev=None, u_limit=None):
        """
        计算控制力矩（与原仿真接口兼容）。

        新反馈律:
            u = (Kp*e - Kd*w_gyro - z3) / b0
        其中:
            e = -q_e[1:4], q_e = quat_error(qd_smooth, q)
        这样可保持与原项目 PD 方向一致（原式为 -Kp*q_e[1:] - Kd*w）。

        参数:
            qd: 目标四元数
            q: 当前四元数
            w: 陀螺仪角速度测量(3,)
            dt: 时间步长
            u_prev: 上一时刻实际施加力矩（建议传入饱和值）
            u_limit: 控制输出限幅（标量或3轴向量）

        返回:
            u_cmd: 控制力矩（若设置 u_limit，则为限幅后）
            q_e:   误差四元数（基于平滑参考 qd_smooth）
        """
        dt_use = float(dt) if dt is not None else self.dt

        qd_smooth, _ = self._td_step(qd, dt_use)

        q_e = quat_error(qd_smooth, q)
        e = -q_e[1:]

        w_gyro = np.asarray(w, dtype=float)

        if u_prev is None:
            u_applied_prev = self.u_applied
        else:
            u_applied_prev = np.asarray(u_prev, dtype=float)
        self.eso_step(e, u_applied_prev, dt_use)

        # Derivative on Measurement: 阻尼项直接使用陀螺仪角速度
        u_raw = (self.kp * e - self.kd * w_gyro - self.z3) / self.b0

        if u_limit is not None:
            u_lim = self._vec3(u_limit)
            u_cmd = np.clip(u_raw, -u_lim, u_lim)
        else:
            u_cmd = u_raw

        # 记录当前“命令/限幅后力矩”，在调用方不传 u_prev 时作为后备反饱和输入
        self.u_applied = np.asarray(u_cmd, dtype=float)

        return u_cmd, q_e

    def update_b0(self, new_b0):
        """
        在线更新控制增益 b0（通常取 b0 ≈ J^{-1}）。
        """
        self.b0 = np.maximum(self._vec3(new_b0), 1e-6)

    def get_disturbance_estimate(self):
        """
        获取 ESO 当前总扰动估计 z3（模型域变量）。
        """
        return self.z3.copy()

    def get_disturbance_estimate_torque(self):
        """
        将 ESO 扰动估计换算为等效力矩估计，便于与动力学中的 dist 对比。
        """
        return (self.z3 / self.b0).copy()

    def reset(self):
        """重置控制器内部状态。"""
        self.z1[:] = 0.0
        self.z2[:] = 0.0
        self.z3[:] = 0.0
        self.w_target[:] = 0.0
        self.qd_smooth = None
        self.u_applied[:] = 0.0


def adrc_torque(qd, q, w, controller, dt, u_prev=None):
    """
    ADRC 控制器接口函数（兼容 pd_torque 调用方式）。
    """
    u, q_e = controller.compute_control(qd, q, w, dt, u_prev)
    return u, q_e
