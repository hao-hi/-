"""
卫星姿态控制仿真主程序
整合动力学、控制器、MEKF、星敏感器等模块
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
from collections import deque
from pathlib import Path
from datetime import datetime
from config import CONFIG
from core_utils import quat_from_axis_angle, quat_to_R, quat_mul, quat_angle_errors_deg, jit
from dynamics import Spacecraft
from controller import pd_torque
from adrc_controller import ADRCController, adrc_torque
from startracker import StarTracker
from mekf import MEKFBiasOnly, MEKF_Augmented
from estimators import InertiaRLS, ensure_diag_inertia
try:
    from 优化器.optimizers import (
        grid_search,
        random_search,
        nelder_mead,
        simulated_annealing,
        pso
    )
except ModuleNotFoundError:
    from optimizers import (
        grid_search,
        random_search,
        nelder_mead,
        simulated_annealing,
        pso
    )
from visualization import (
    plot_attitude_estimation,
    plot_attitude_error,
    plot_angular_velocity,
    plot_control_torque,
    plot_control_response,
    plot_observer_tracking,
    plot_controller_comparison_dashboard,
    plot_simulation_process_overview,
    plot_inertia_identification,
    plot_inertia_identification_dashboard,
    plot_gain_landscape_from_cache,
    plot_optimizer_report_dashboard,
    plot_optimizer_convergence_statistics,
    plot_optimizer_best_response_dashboard,
    plot_optimizer_tradeoff_scatter,
    plot_optimizer_metric_heatmap,
    plot_multiple_responses,
    plot_simulation_report_dashboard
)

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DETAIL_LEVEL = "core"
_TRAPEZOID = getattr(np, "trapezoid", np.trapz)
IDENTIFICATION_PROBE_PROFILE = {
    'probe_freqs': [0.32, 0.51, 0.79, 0.46, 0.68, 0.95],
    'probe_phases': [0.0, 0.9, 1.7, 1.2, 0.4, 2.1],
}


def _make_output_layout(root_dir):
    """
    创建统一的输出目录结构。
    """
    root = Path(root_dir)
    layout = {
        'root': root,
        'optimization': root / 'optimization',
        'optimization_adrc': root / 'optimization' / 'adrc',
        'comparison': root / 'comparison',
        'identification': root / 'identification',
        'identification_rls': root / 'identification' / 'rls',
        'identification_mekf': root / 'identification' / 'mekf',
        'simulations': root / 'simulations',
        'simulation_pd': root / 'simulations' / 'pd',
        'simulation_adrc': root / 'simulations' / 'adrc',
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def _save_figure(fig, path, dpi=200):
    """
    统一保存图像，确保目录存在并在保存后关闭 figure。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def _write_csv_rows(path, fieldnames, rows):
    """
    以 UTF-8 BOM 编码写入 CSV，便于在中文环境中直接打开。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _markdown_table(headers, rows):
    """
    构造 Markdown 表格字符串。
    """
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(['---'] * len(headers)) + " |"
    body_lines = []
    for row in rows:
        body_lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join([header_line, sep_line, *body_lines])


def _relative_markdown_path(base_dir, target_path):
    """
    生成适合写入 Markdown 的相对路径。
    """
    return Path(target_path).relative_to(base_dir).as_posix()


def _blend_vector(previous, current, alpha):
    """
    对向量序列做一阶指数平滑，抑制差分量测抖动。
    """
    current = np.asarray(current, dtype=float)
    if previous is None:
        return current.copy()
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * current + (1.0 - alpha) * np.asarray(previous, dtype=float)


def _curve_integral(y, x):
    """
    统一的梯形积分兼容层，适配新旧 NumPy。
    """
    return float(_TRAPEZOID(y, x))


def _build_augmented_covariance_blocks(P_j_scale, Q_j_scale):
    """
    构造增广 MEKF 的初始协方差与过程噪声块矩阵。
    """
    P_j_scale = float(P_j_scale)
    Q_j_scale = float(Q_j_scale)
    return (
        np.block([
            [1e-4 * np.eye(3), np.zeros((3, 3)), np.zeros((3, 3))],
            [np.zeros((3, 3)), 1e-6 * np.eye(3), np.zeros((3, 3))],
            [np.zeros((3, 3)), np.zeros((3, 3)), P_j_scale * np.eye(3)],
        ]),
        np.block([
            [1e-7 * np.eye(3), np.zeros((3, 3)), np.zeros((3, 3))],
            [np.zeros((3, 3)), 1e-10 * np.eye(3), np.zeros((3, 3))],
            [np.zeros((3, 3)), np.zeros((3, 3)), Q_j_scale * np.eye(3)],
        ]),
    )


def _with_identification_probe(cfg, probe_amp, probe_duration):
    """
    为辨识配置补齐统一的探测力矩参数。
    """
    merged = dict(cfg)
    merged.update(IDENTIFICATION_PROBE_PROFILE)
    probe_amp_arr = np.asarray(probe_amp, dtype=float).reshape(-1)
    merged['probe_amp'] = float(probe_amp_arr[0]) if probe_amp_arr.size == 1 else probe_amp_arr[:3].copy()
    merged['probe_duration'] = float(probe_duration)
    return merged


def _build_identification_case_configs():
    """
    统一构造前置惯量辨识案例的 RLS / MEKF 配置。
    """
    mekf_P0, mekf_Q = _build_augmented_covariance_blocks(P_j_scale=2.5e-1, Q_j_scale=1.8e-7)
    rls_cfg = _with_identification_probe(
        {
            'scheme': 'RLS',
            'J0': np.array([1.45, 1.30, 0.95], dtype=float),
            'lambda_factor': 1.0,
            'P0': 2.5 * np.eye(3, dtype=float),
            'min_inertia': 0.50,
            'max_inertia': 2.40,
            'min_accel_excitation': 0.008,
            'min_torque_excitation': 0.012,
            'max_update_step': 2.5e-2,
            'min_regressor_norm': 0.005,
            'innovation_clip': 0.06,
            'innovation_deadzone': 0.0025,
            'min_update_norm': 4e-5,
            'regularization': 1e-6,
            'covariance_ceiling': 20.0,
            'excitation_alpha': 0.08,
            'warmup_time': 1.20,
            'wdot_filter_alpha': 1.0,
            'wdot_regression_window': 2,
            'disturbance_filter_alpha': 0.15,
            'disturbance_clip': 0.02,
            'disturbance_enable_time': 2.0,
            'window_size': 12,
            'theta_smoothing': 0.90,
            'filter_alpha': 0.18,
        },
        probe_amp=np.array([0.18, 0.20, 0.10], dtype=float),
        probe_duration=14.0,
    )
    mekf_cfg = _with_identification_probe(
        {
            'scheme': 'MEKF',
            'J0': np.array([1.25, 1.15, 0.82], dtype=float),
            'P0': mekf_P0,
            'Q': mekf_Q,
            'min_inertia': 0.50,
            'max_inertia': 2.40,
            'min_regressor_norm': 2.5e-3,
            'max_inertia_step': 8.0e-3,
            'dynamics_measurement_noise': 0.06,
            'min_dynamics_excitation': 0.03,
            'dynamics_innovation_clip': 0.30,
            'inertia_update_gain': 0.95,
            'covariance_ceiling': 0.5,
            'wdot_filter_alpha': 0.65,
            'wdot_regression_window': 4,
            'disturbance_filter_alpha': 0.18,
            'disturbance_clip': 0.03,
            'disturbance_enable_time': 1.5,
        },
        probe_amp=0.16,
        probe_duration=8.0,
    )
    return {
        'RLS': rls_cfg,
        'MEKF': mekf_cfg,
    }


def _resolve_excitation_initial_state(rng, T, excitation_profile, rate_scale=1.0):
    """
    统一生成初始姿态与角速度，保证不同仿真路径使用相同的激励规则。
    """
    axis = rng.randn(3)
    axis /= np.linalg.norm(axis)

    excitation_mode = str(excitation_profile).strip().lower()
    if excitation_mode == 'auto':
        excitation_mode = 'aggressive' if float(T) > 15.0 else 'nominal'

    if excitation_mode == 'aggressive':
        initial_angle = np.deg2rad(45.0)
        initial_rate_deg = np.array([1.0, -1.0, 0.5], dtype=float)
    else:
        initial_angle = np.deg2rad(15.0)
        initial_rate_deg = np.array([0.6, -0.6, 0.3], dtype=float)

    q0 = quat_from_axis_angle(axis, initial_angle)
    w0 = np.deg2rad(initial_rate_deg * float(np.clip(rate_scale, 0.5, 8.0)))
    return q0, w0


def _prepare_random_batches(rng, steps, star_noise_std):
    """
    批量生成主循环中的随机量，降低循环内的 Python 调用开销。
    """
    return {
        'fallback_noise_axes': rng.randn(steps, 3),
        'fallback_startracker_angles': np.abs(rng.normal(0.0, star_noise_std, size=steps)),
        'fallback_ideal_angles': np.abs(rng.normal(0.0, 0.001, size=steps)),
        'gyro_bias_rw_noise': rng.randn(steps, 3),
        'gyro_measurement_noise': rng.randn(steps, 3),
        'dist_noise_samples': rng.randn(steps, 3),
    }


@jit(nopython=True)
def _simulate_pd_metrics_kernel(Kp, Kd, dt, umax, q0, w0, J_diag, dist_const, dist_noise_std, dist_noise_samples):
    """
    面向 PD 自动调参的专用数值内核。

    该内核只保留轻量指标仿真所需的最小计算集合，避免在热路径中频繁跨 Python
    层调用通用控制器/动力学封装。
    """
    steps = dist_noise_samples.shape[0]
    err_hist = np.empty(steps, dtype=np.float64)
    u_norm_hist = np.empty(steps, dtype=np.float64)
    sat_count = 0

    q = np.empty(4, dtype=np.float64)
    w = np.empty(3, dtype=np.float64)
    q[:] = q0
    w[:] = w0

    invJ = 1.0 / np.maximum(J_diag, 1e-12)

    def quat_norm_local(q_in):
        return q_in / np.sqrt(np.dot(q_in, q_in) + 1e-15)

    def quat_dot_local(q_in, w_in):
        wx, wy, wz = w_in
        qdot = np.empty(4, dtype=np.float64)
        qdot[0] = 0.5 * (-wx * q_in[1] - wy * q_in[2] - wz * q_in[3])
        qdot[1] = 0.5 * ( wx * q_in[0] + wz * q_in[2] - wy * q_in[3])
        qdot[2] = 0.5 * ( wy * q_in[0] - wz * q_in[1] + wx * q_in[3])
        qdot[3] = 0.5 * ( wz * q_in[0] + wy * q_in[1] - wx * q_in[2])
        return qdot

    def omega_dot_local(w_in, u_sat_in, dist_in):
        wx, wy, wz = w_in
        Jx, Jy, Jz = J_diag
        cross_x = (Jz - Jy) * wy * wz
        cross_y = (Jx - Jz) * wx * wz
        cross_z = (Jy - Jx) * wx * wy
        out = np.empty(3, dtype=np.float64)
        out[0] = (u_sat_in[0] + dist_in[0] - cross_x) * invJ[0]
        out[1] = (u_sat_in[1] + dist_in[1] - cross_y) * invJ[1]
        out[2] = (u_sat_in[2] + dist_in[2] - cross_z) * invJ[2]
        return out

    for k in range(steps):
        sign = 1.0 if q[0] >= 0.0 else -1.0
        u = np.empty(3, dtype=np.float64)
        u[0] = -Kp * sign * q[1] - Kd * w[0]
        u[1] = -Kp * sign * q[2] - Kd * w[1]
        u[2] = -Kp * sign * q[3] - Kd * w[2]

        u_sat = np.empty(3, dtype=np.float64)
        for i in range(3):
            val = u[i]
            if val > umax:
                u_sat[i] = umax
            elif val < -umax:
                u_sat[i] = -umax
            else:
                u_sat[i] = val
        if (
            np.abs(u_sat[0]) >= (umax - 1e-6)
            or np.abs(u_sat[1]) >= (umax - 1e-6)
            or np.abs(u_sat[2]) >= (umax - 1e-6)
        ):
            sat_count += 1

        dist = dist_const + dist_noise_std * dist_noise_samples[k]

        k1_w = omega_dot_local(w, u_sat, dist)
        k1_q = quat_dot_local(q, w)

        w_temp = w + 0.5 * dt * k1_w
        q_temp = quat_norm_local(q + 0.5 * dt * k1_q)
        k2_w = omega_dot_local(w_temp, u_sat, dist)
        k2_q = quat_dot_local(q_temp, w_temp)

        w_temp = w + 0.5 * dt * k2_w
        q_temp = quat_norm_local(q + 0.5 * dt * k2_q)
        k3_w = omega_dot_local(w_temp, u_sat, dist)
        k3_q = quat_dot_local(q_temp, w_temp)

        w_temp = w + dt * k3_w
        q_temp = quat_norm_local(q + dt * k3_q)
        k4_w = omega_dot_local(w_temp, u_sat, dist)
        k4_q = quat_dot_local(q_temp, w_temp)

        w = w + (dt / 6.0) * (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w)
        q = quat_norm_local(q + (dt / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q))

        q0_abs = np.abs(q[0])
        if q0_abs > 1.0:
            q0_abs = 1.0
        err_hist[k] = np.rad2deg(2.0 * np.arccos(q0_abs))
        u_norm_hist[k] = np.sqrt(np.dot(u_sat, u_sat))

    return err_hist, u_norm_hist, sat_count


def _resolve_run_profile(profile):
    """
    统一管理项目运行档位，便于在速度与精度之间快速切换。
    """
    profile_name = str(profile).strip().lower()
    if profile_name == 'fast':
        return {
            'profile': 'fast',
            'T': 8.0,
            'dt': 0.04,
            'optimizer_quick': True,
            'optimizer_runs': 1,
            'objective_eval_seeds': [0],
            'use_star_tracker': True,
            'save_comparison_plots': False,
        }
    return {
        'profile': 'full',
        'T': 15.0,
        'dt': 0.03,
        'optimizer_quick': False,
        'optimizer_runs': 2,
        'objective_eval_seeds': [0, 1],
        'use_star_tracker': True,
        'save_comparison_plots': True,
    }


def _build_inertia_identifier(inertia_estimator_cfg, initial_J_diag, initial_q):
    """
    创建在线惯量辨识器，返回 (scheme, estimator_or_filter)。
    """
    if inertia_estimator_cfg is None:
        return None, None

    cfg = dict(inertia_estimator_cfg)
    scheme = str(cfg.get('scheme', '')).strip().upper()
    J0 = ensure_diag_inertia(cfg.get('J0', initial_J_diag))

    if scheme == 'RLS':
        estimator = InertiaRLS(
            J0=J0,
            lambda_factor=cfg.get('lambda_factor', 0.98),
            P0=cfg.get('P0'),
            min_inertia=cfg.get('min_inertia', 1e-4),
            max_inertia=cfg.get('max_inertia', 5.0),
            min_accel_excitation=cfg.get('min_accel_excitation', 1e-3),
            min_torque_excitation=cfg.get('min_torque_excitation', 1e-3),
            max_update_step=cfg.get('max_update_step', 5e-3),
            min_regressor_norm=cfg.get('min_regressor_norm', 1e-3),
            innovation_clip=cfg.get('innovation_clip', 0.08),
            regularization=cfg.get('regularization', 1e-8),
            covariance_floor=cfg.get('covariance_floor', 1e-6),
            covariance_ceiling=cfg.get('covariance_ceiling', 50.0),
            excitation_alpha=cfg.get('excitation_alpha', 0.10),
            innovation_deadzone=cfg.get('innovation_deadzone', 0.0),
            min_update_norm=cfg.get('min_update_norm', 0.0),
            window_size=cfg.get('window_size', 8),
            theta_smoothing=cfg.get('theta_smoothing', 0.60),
            filter_alpha=cfg.get('filter_alpha', 0.15),
            axis_weight_floor=cfg.get('axis_weight_floor', 0.20),
            axis_weight_power=cfg.get('axis_weight_power', 0.50),
        )
        return scheme, estimator

    if scheme == 'MEKF':
        estimator = MEKF_Augmented(
            q0=initial_q,
            J0=J0,
            P0=cfg.get('P0'),
            Q=cfg.get('Q'),
            R=cfg.get('R'),
            inertia_bounds=(
                cfg.get('min_inertia', 1e-4),
                cfg.get('max_inertia', 5.0),
            ),
            dynamics_measurement_noise=cfg.get('dynamics_measurement_noise', 0.15),
            min_dynamics_excitation=cfg.get('min_dynamics_excitation', 0.05),
            min_regressor_norm=cfg.get('min_regressor_norm', 2e-3),
            dynamics_innovation_clip=cfg.get('dynamics_innovation_clip', 0.6),
            max_inertia_step=cfg.get('max_inertia_step', 8e-4),
            inertia_update_gain=cfg.get('inertia_update_gain', 0.65),
            covariance_floor=cfg.get('covariance_floor', 1e-9),
            covariance_ceiling=cfg.get('covariance_ceiling', 1.0),
        )
        return scheme, estimator

    raise ValueError("inertia_estimator_cfg['scheme'] 仅支持 'RLS' 或 'MEKF'")


def _sync_inertia_estimate(J_diag, adrc_controller):
    """
    将最新惯量估计同步给控制器内部模型。

    注意:
        真实动力学模型在仿真中应保持固定，不能把在线辨识结果直接写回“被控对象”，
        否则会把参数辨识问题变成“边辨识边修改真实系统”，导致结果失真。
    """
    J_diag = ensure_diag_inertia(J_diag)
    if adrc_controller is not None:
        adrc_controller.update_b0(1.0 / np.maximum(J_diag, 1e-6))
    return J_diag


def _build_default_adrc_params(dt, estimated_inertia=None, omega_c=4.0, omega_o=8.0):
    """
    构造默认 ADRC 参数，并按估计惯量匹配 b0。
    """
    if estimated_inertia is None:
        estimated_inertia = np.array([0.50, 0.50, 0.50], dtype=float)
    else:
        estimated_inertia = ensure_diag_inertia(estimated_inertia)
    return {
        'b0': 1.0 / estimated_inertia,
        'omega_c': float(omega_c),
        'omega_o': float(omega_o),
        'dt': dt,
    }


def _init_controller(controller_kind, dt, Kp, Kd, adrc_params=None, verbose=True):
    """
    统一初始化控制器实例，并返回解析后的参数。
    """
    controller_tag = str(controller_kind).upper()
    if controller_tag == 'ADRC':
        resolved_adrc_params = _build_default_adrc_params(dt) if adrc_params is None else dict(adrc_params)
        controller = ADRCController(**resolved_adrc_params)
        if verbose:
            print(
                f"使用ADRC控制器: "
                f"omega_c={resolved_adrc_params.get('omega_c', 2.5)}, "
                f"omega_o={resolved_adrc_params.get('omega_o', 8.0)}"
            )
        return controller, resolved_adrc_params

    if verbose:
        print(f"使用PD控制器: Kp={Kp}, Kd={Kd}")
    return None, None


def _resolve_true_inertia_matrix(true_inertia):
    """
    将真实转动惯量输入规范化为 3x3 对角矩阵。
    """
    if true_inertia is None:
        return np.asarray(CONFIG['spacecraft']['inertia'], dtype=float)
    return np.diag(ensure_diag_inertia(true_inertia))


def _allocate_simulation_history(steps):
    """
    统一分配仿真历史数组，减少主循环前的样板代码。
    """
    return {
        't': np.empty(steps, dtype=float),
        'q_true': np.empty((steps, 4), dtype=float),
        'q_est': np.empty((steps, 4), dtype=float),
        'w': np.empty((steps, 3), dtype=float),
        'u': np.empty((steps, 3), dtype=float),
        'err': np.empty(steps, dtype=float),
        'dist_true': np.empty((steps, 3), dtype=float),
        'dist_est': np.empty((steps, 3), dtype=float),
        'gyro_bias_true': np.empty((steps, 3), dtype=float),
        'gyro_bias_est': np.empty((steps, 3), dtype=float),
        'inertia_est': np.empty((steps, 3), dtype=float),
        'inertia_update_mask': np.empty(steps, dtype=float),
        'wdot_est': np.empty((steps, 3), dtype=float),
        'regressor_min_sv': np.full(steps, np.nan, dtype=float),
        'axis_regressor_norms': np.full((steps, 3), np.nan, dtype=float),
    }


def _build_estimation_context(sc, inertia_estimator_cfg, initial_q, dt):
    """
    统一初始化 MEKF 与在线惯量辨识相关上下文。
    """
    initial_J_diag = np.diag(sc.J).copy()
    inertia_scheme, inertia_identifier = _build_inertia_identifier(
        inertia_estimator_cfg,
        initial_J_diag=initial_J_diag,
        initial_q=initial_q,
    )
    if inertia_scheme == 'MEKF':
        mekf = inertia_identifier
    else:
        mekf = MEKFBiasOnly()
        mekf.q = initial_q.copy()

    estimator_cfg = {} if inertia_estimator_cfg is None else dict(inertia_estimator_cfg)
    use_augmented_mekf = isinstance(mekf, MEKF_Augmented)
    regressor_min_sv_reference = None
    if inertia_scheme == 'RLS' and inertia_identifier is not None:
        regressor_min_sv_reference = float(
            estimator_cfg.get('min_regressor_norm', inertia_identifier.min_regressor_norm)
        )

    if inertia_scheme == 'RLS' and inertia_identifier is not None:
        current_J_diag = inertia_identifier.get_inertia_diag()
    elif inertia_scheme == 'MEKF' and use_augmented_mekf:
        current_J_diag = mekf.get_inertia_diag()
    else:
        current_J_diag = initial_J_diag.copy()

    return {
        'mekf': mekf,
        'use_augmented_mekf': use_augmented_mekf,
        'inertia_scheme': inertia_scheme,
        'inertia_identifier': inertia_identifier,
        'inertia_reference': initial_J_diag.copy(),
        'nominal_J_diag': initial_J_diag.copy(),
        'current_J_diag': current_J_diag,
        'estimator_cfg': estimator_cfg,
        'inertia_warmup_steps': int(max(0.0, float(estimator_cfg.get('warmup_time', 0.0))) / dt),
        'wdot_filter_alpha': float(np.clip(estimator_cfg.get('wdot_filter_alpha', 1.0), 0.0, 1.0)),
        'wdot_regression_window': max(2, int(estimator_cfg.get('wdot_regression_window', 2))),
        'disturbance_filter_alpha': float(np.clip(estimator_cfg.get('disturbance_filter_alpha', 1.0), 0.0, 1.0)),
        'disturbance_enable_steps': int(max(0.0, float(estimator_cfg.get('disturbance_enable_time', 0.0))) / dt),
        'disturbance_clip': estimator_cfg.get('disturbance_clip'),
        'regressor_min_sv_reference': regressor_min_sv_reference,
    }


def _apply_quaternion_noise(q, noise_axis, noise_angle):
    """
    给四元数姿态测量注入轴角噪声。
    """
    axis = np.asarray(noise_axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    noise_quat = quat_from_axis_angle(axis, float(noise_angle))
    q_noisy = quat_mul(np.asarray(q, dtype=float), noise_quat)
    return q_noisy / np.linalg.norm(q_noisy)


def _compute_identification_probe_torque(t, estimator_cfg, umax):
    """
    生成用于参数辨识的轻量多频探测力矩。
    """
    if estimator_cfg is None:
        return np.zeros(3, dtype=float)

    probe_amp = estimator_cfg.get('probe_amp')
    probe_duration = float(max(0.0, estimator_cfg.get('probe_duration', 0.0)))
    if probe_amp is None or probe_duration <= 0.0 or t > probe_duration:
        return np.zeros(3, dtype=float)

    amp_arr = np.asarray(probe_amp, dtype=float).reshape(-1)
    if amp_arr.size == 1:
        amp_vec = np.full(3, float(amp_arr[0]), dtype=float)
    else:
        amp_vec = np.pad(amp_arr[:3], (0, max(0, 3 - amp_arr[:3].size)), mode='edge').astype(float)

    if np.max(np.abs(amp_vec)) <= 1.0:
        amp_vec = amp_vec * float(umax)

    freqs = np.asarray(estimator_cfg.get('probe_freqs', [0.35, 0.57, 0.83, 0.48, 0.71, 0.93]), dtype=float).reshape(-1)
    phases = np.asarray(estimator_cfg.get('probe_phases', [0.0, 0.8, 1.6, 1.1, 0.3, 2.0]), dtype=float).reshape(-1)
    if freqs.size < 6:
        freqs = np.pad(freqs, (0, 6 - freqs.size), mode='edge')
    if phases.size < 6:
        phases = np.pad(phases, (0, 6 - phases.size), mode='edge')

    ramp_time = max(0.6, 0.15 * probe_duration)
    ramp = min(1.0, max(0.0, t / ramp_time))
    primary = np.sin(2.0 * np.pi * freqs[:3] * t + phases[:3])
    secondary = np.cos(2.0 * np.pi * freqs[3:6] * t + phases[3:6])
    probe_shape = 0.72 * primary + 0.28 * secondary
    probe = ramp * amp_vec * probe_shape
    return probe.astype(float)


def _resolve_attitude_measurement(
    q,
    use_star_tracker,
    star_tracker,
    step_idx,
    fallback_noise_axes,
    fallback_startracker_angles,
    fallback_ideal_angles,
):
    """
    统一处理星敏感器观测与退化时的噪声姿态测量。
    """
    if use_star_tracker and star_tracker is not None:
        q_meas = star_tracker.observe(quat_to_R(q))
        if q_meas is not None:
            return q_meas
        return _apply_quaternion_noise(q, fallback_noise_axes[step_idx], fallback_startracker_angles[step_idx])
    return _apply_quaternion_noise(q, fallback_noise_axes[step_idx], fallback_ideal_angles[step_idx])


def _estimate_angular_rate_regression_state(w_samples, dt):
    """
    使用短窗线性回归同时估计局部平均角速度与角加速度。

    相比简单两点差分，这种做法对陀螺噪声更稳，也更适合在线惯量辨识。
    """
    if not w_samples:
        return np.zeros(3, dtype=float), np.zeros(3, dtype=float)

    W = np.vstack(w_samples).astype(float, copy=False)
    omega_reg = np.mean(W, axis=0)
    if W.shape[0] < 2:
        return omega_reg, np.zeros(3, dtype=float)

    idx = np.arange(W.shape[0], dtype=float)
    idx_center = idx - np.mean(idx)
    denom = float(np.sum(idx_center ** 2))
    if denom <= 1e-12:
        return omega_reg, np.zeros(3, dtype=float)

    slope = (idx_center[:, None] * W).sum(axis=0) / (denom * float(dt))
    return omega_reg, slope


def _compute_settle_time(err_hist, t_hist, strict_threshold=2.0, trans_threshold=3.0):
    """
    采用两阶段规则计算调节时间。
    """
    err = np.asarray(err_hist, dtype=float)
    t_arr = np.asarray(t_hist, dtype=float)
    settle_time = t_arr[-1] + 10.0
    stable_window = max(20, int(err.size * 0.05))

    if err.size >= stable_window:
        strict_mask = (err <= strict_threshold).astype(np.int8, copy=False)
        stable_counts = np.convolve(
            strict_mask,
            np.ones(stable_window, dtype=np.int16),
            mode='valid',
        )
        stable_idx = np.flatnonzero(stable_counts == stable_window)
        if stable_idx.size:
            return float(t_arr[int(stable_idx[0])])

    trans_idx = np.flatnonzero(err > trans_threshold)
    if trans_idx.size:
        return float(t_arr[int(trans_idx[-1])] + 1.0)

    return float(settle_time)


def _compute_overshoot(err_hist):
    """
    计算姿态误差调节过程中的超调量。

    当前误差定义为“姿态角误差模值”，且仿真从非零初始误差收敛到零。
    因此超调不应把初始误差本身计入，而应只统计控制启动后
    是否出现“超过初始误差峰值”的二次抬升。
    """
    err = np.asarray(err_hist, dtype=float).reshape(-1)
    if err.size == 0:
        return 0.0
    if err.size == 1:
        return 0.0

    initial_error = float(err[0])
    peak_after_start = float(np.max(err[1:]))
    return float(max(0.0, peak_after_start - initial_error))


def _tail_window_slice(length, tail_ratio=0.15, min_samples=8):
    """
    返回结果尾段切片，用于稳态统计。
    """
    n = int(max(0, length))
    if n <= 0:
        return slice(0, 0)
    window = max(int(min_samples), int(np.ceil(n * float(tail_ratio))))
    window = min(n, max(1, window))
    return slice(n - window, n)


def _resolve_disturbance_torque_estimate(
    step_idx,
    inertia_scheme,
    inertia_identifier,
    adrc_controller,
    disturbance_filter_alpha,
    disturbance_clip,
    disturbance_enable_steps,
    dist_torque_filt_prev,
):
    """
    统一计算供在线辨识使用的扰动力矩估计。
    """
    if inertia_identifier is None or inertia_scheme not in {'RLS', 'MEKF'}:
        return np.zeros(3, dtype=float), dist_torque_filt_prev

    dist_torque_raw = (
        adrc_controller.get_disturbance_estimate_torque()
        if adrc_controller is not None else np.zeros(3, dtype=float)
    )
    dist_torque_est = _blend_vector(dist_torque_filt_prev, dist_torque_raw, disturbance_filter_alpha)
    next_prev = dist_torque_est.copy()

    if disturbance_clip is not None:
        dist_norm = np.linalg.norm(dist_torque_est)
        dist_limit = float(max(0.0, disturbance_clip))
        if dist_norm > dist_limit > 0.0:
            dist_torque_est = dist_torque_est * (dist_limit / (dist_norm + 1e-12))
    if step_idx < disturbance_enable_steps:
        dist_torque_est = np.zeros(3, dtype=float)

    return dist_torque_est, next_prev


def _update_inertia_estimation(
    step_idx,
    inertia_scheme,
    inertia_identifier,
    use_augmented_mekf,
    mekf,
    inertia_warmup_steps,
    omega_id,
    wdot_est,
    u_prev,
    dist_torque_est,
    nominal_J_diag,
    adrc_controller,
):
    """
    统一处理 RLS / 增广 MEKF 的惯量更新与诊断量输出。
    """
    if inertia_scheme == 'RLS' and inertia_identifier is not None and step_idx >= inertia_warmup_steps:
        J_diag_est, inertia_updated = inertia_identifier.update(
            omega=omega_id,
            wdot=wdot_est,
            u=u_prev,
            disturbance_torque=dist_torque_est,
        )
        return (
            _sync_inertia_estimate(J_diag_est, adrc_controller),
            inertia_updated,
            inertia_identifier.get_last_regressor_min_sv(),
            inertia_identifier.get_last_axis_regressor_norms(),
        )

    if inertia_scheme == 'RLS' and inertia_identifier is not None:
        return (
            _sync_inertia_estimate(inertia_identifier.get_inertia_diag(), adrc_controller),
            False,
            inertia_identifier.get_last_regressor_min_sv(),
            inertia_identifier.get_last_axis_regressor_norms(),
        )

    if inertia_scheme == 'MEKF' and use_augmented_mekf:
        inertia_updated = (
            mekf.update_dynamics(wdot_est, disturbance_torque=dist_torque_est)
            if step_idx >= inertia_warmup_steps else False
        )
        return (
            _sync_inertia_estimate(mekf.get_inertia_diag(), adrc_controller),
            inertia_updated,
            np.nan,
            np.full(3, np.nan, dtype=float),
        )

    if inertia_scheme == 'MEKF' and inertia_identifier is not None:
        return (
            _sync_inertia_estimate(mekf.get_inertia_diag(), adrc_controller),
            False,
            np.nan,
            np.full(3, np.nan, dtype=float),
        )

    return nominal_J_diag.copy(), False, np.nan, np.full(3, np.nan, dtype=float)


def _compute_control_command(controller_kind, qd, q, w, Kp, Kd, adrc_controller, dt, u_prev):
    """
    统一计算当前控制输入及其扰动估计输出。
    """
    if str(controller_kind).upper() == 'ADRC':
        u_cmd, _ = adrc_torque(qd, q, w, adrc_controller, dt, u_prev)
        return u_cmd, adrc_controller.get_disturbance_estimate_torque()

    u_cmd, _ = pd_torque(qd, q, w, Kp, Kd)
    return u_cmd, np.zeros(3, dtype=float)


def _resolve_control_feedback_state(control_feedback_source, q_true, w_true, q_est, w_est):
    """
    统一决定控制器使用真实状态还是估计状态闭环。
    """
    mode = str(control_feedback_source).strip().lower()
    if mode == 'estimate':
        return (
            np.asarray(q_est, dtype=float).copy(),
            np.asarray(w_est, dtype=float).copy(),
            mode,
        )
    if mode == 'truth':
        return (
            np.asarray(q_true, dtype=float).copy(),
            np.asarray(w_true, dtype=float).copy(),
            mode,
        )
    raise ValueError("control_feedback_source must be 'truth' or 'estimate'")


def _record_simulation_sample(
    history,
    step_idx,
    t,
    q,
    q_est,
    w,
    u_sat,
    err_deg,
    dist_true,
    dist_est,
    gyro_bias_true,
    gyro_bias_est,
    current_J_diag,
    inertia_updated,
    wdot_est,
    regressor_min_sv,
    axis_regressor_norms,
):
    """
    将当前时刻的仿真状态统一写入历史数组。
    """
    history['t'][step_idx] = t
    history['q_true'][step_idx] = q
    history['q_est'][step_idx] = q_est
    history['w'][step_idx] = w
    history['u'][step_idx] = u_sat
    history['err'][step_idx] = err_deg
    history['dist_true'][step_idx] = dist_true
    history['dist_est'][step_idx] = dist_est
    history['gyro_bias_true'][step_idx] = gyro_bias_true
    history['gyro_bias_est'][step_idx] = gyro_bias_est
    history['inertia_est'][step_idx] = current_J_diag
    history['inertia_update_mask'][step_idx] = 1.0 if inertia_updated else 0.0
    history['wdot_est'][step_idx] = wdot_est
    history['regressor_min_sv'][step_idx] = regressor_min_sv
    history['axis_regressor_norms'][step_idx] = axis_regressor_norms


def _compute_control_metrics(u_hist, umax):
    """
    统一计算控制输入的范数、单轴峰值和限幅利用率指标。
    """
    u_arr = np.asarray(u_hist, dtype=float)
    if u_arr.ndim != 2 or u_arr.shape[1] != 3:
        raise ValueError("u_hist 必须是 N×3 控制力矩数组")

    if u_arr.shape[0] == 0:
        return {
            'u_norm': np.empty(0, dtype=float),
            'u_axis_peak': np.empty(0, dtype=float),
            'peak_torque_norm': 0.0,
            'peak_axis_torque': 0.0,
            'control_limit_norm': float(np.sqrt(3.0) * float(umax)),
            'torque_usage_ratio': 0.0,
        }

    u_norm = np.linalg.norm(u_arr, axis=1)
    u_axis_peak = np.max(np.abs(u_arr), axis=1)
    peak_axis_torque = float(np.max(u_axis_peak))
    return {
        'u_norm': u_norm,
        'u_axis_peak': u_axis_peak,
        'peak_torque_norm': float(np.max(u_norm)),
        'peak_axis_torque': peak_axis_torque,
        'control_limit_norm': float(np.sqrt(u_arr.shape[1]) * float(umax)),
        'torque_usage_ratio': peak_axis_torque / max(float(umax), 1e-12),
    }


def _derive_simulation_metrics(history, settle_time, sat_count, umax):
    """
    从仿真历史统一派生结果指标，供表格、报告和图表复用。
    """
    err_hist = np.asarray(history['err'], dtype=float)
    t_hist = np.asarray(history['t'], dtype=float)
    u_hist = np.asarray(history['u'], dtype=float)
    w_hist = np.asarray(history['w'], dtype=float)
    control_metrics = _compute_control_metrics(u_hist, umax)

    overshoot = _compute_overshoot(err_hist)
    final_error = float(err_hist[-1]) if err_hist.size else 0.0
    abs_err = np.abs(err_hist)
    u_norm = control_metrics['u_norm']
    w_norm_deg = np.rad2deg(np.linalg.norm(w_hist, axis=1))
    tail_slice = _tail_window_slice(err_hist.size)
    tail_err = err_hist[tail_slice]

    attitude_est_rmse = np.nan
    if history.get('q_true') is not None and history.get('q_est') is not None:
        q_err_deg = quat_angle_errors_deg(history['q_true'], history['q_est'])
        attitude_est_rmse = float(np.sqrt(np.mean(q_err_deg ** 2)))

    bias_rmse = np.nan
    if history.get('gyro_bias_true') is not None and history.get('gyro_bias_est') is not None:
        bias_delta = history['gyro_bias_true'] - history['gyro_bias_est']
        bias_rmse = float(np.sqrt(np.mean(np.sum(bias_delta ** 2, axis=1))))

    disturbance_rmse = np.nan
    if history.get('dist_true') is not None and history.get('dist_est') is not None:
        dist_delta = history['dist_true'] - history['dist_est']
        disturbance_rmse = float(np.sqrt(np.mean(np.sum(dist_delta ** 2, axis=1))))

    return {
        'settle_time': float(settle_time),
        'overshoot': float(overshoot),
        'final_error': final_error,
        'peak_error': float(np.max(err_hist)) if err_hist.size else 0.0,
        'rms_error': float(np.sqrt(np.mean(err_hist ** 2))) if err_hist.size else 0.0,
        'steady_state_rms': float(np.sqrt(np.mean(tail_err ** 2))) if tail_err.size else final_error,
        'steady_state_mean': float(np.mean(tail_err)) if tail_err.size else final_error,
        'effort': _curve_integral(u_norm, t_hist) if t_hist.size else 0.0,
        'IAE': _curve_integral(abs_err, t_hist) if t_hist.size else 0.0,
        'ITAE': _curve_integral(t_hist * abs_err, t_hist) if t_hist.size else 0.0,
        'ISU': _curve_integral(np.sum(u_hist ** 2, axis=1), t_hist) if t_hist.size else 0.0,
        'peak_rate_deg': float(np.max(w_norm_deg)) if w_norm_deg.size else 0.0,
        'peak_torque': control_metrics['peak_torque_norm'],
        'peak_axis_torque': control_metrics['peak_axis_torque'],
        'peak_torque_norm': control_metrics['peak_torque_norm'],
        'control_limit_norm': control_metrics['control_limit_norm'],
        'torque_usage_ratio': control_metrics['torque_usage_ratio'],
        'sat_ratio': float(sat_count) / max(1, len(t_hist)),
        'attitude_est_rmse': attitude_est_rmse,
        'bias_rmse': bias_rmse,
        'disturbance_rmse': disturbance_rmse,
    }


def _assemble_simulation_results(
    history,
    dist_const,
    settle_time,
    sat_count,
    umax,
    controller_kind,
    controller_type,
    Kp,
    Kd,
    inertia_scheme,
    control_feedback_source,
    inertia_reference=None,
    regressor_min_sv_reference=None,
):
    """
    汇总仿真历史与性能指标。
    """
    results = dict(history)
    results.update(_derive_simulation_metrics(history, settle_time, sat_count, umax))
    results.update({
        'dist_const': dist_const.copy(),
        'inertia_estimator_scheme': inertia_scheme,
        'Kp': Kp if controller_kind == 'PD' else None,
        'Kd': Kd if controller_kind == 'PD' else None,
        'controller_type': controller_type,
        'control_feedback_source': str(control_feedback_source).strip().lower(),
        'u_limit': float(umax),
        'inertia_reference': None if inertia_reference is None else ensure_diag_inertia(inertia_reference),
        'regressor_min_sv_reference': None if regressor_min_sv_reference is None else float(regressor_min_sv_reference),
    })
    return results


def _build_simulation_figures(results, umax, controller_kind, detail_level='core'):
    """
    构建单次仿真的完整图像集合，便于显示或批量保存。
    """
    controller_tag = str(controller_kind).upper()
    controller_name = 'PD控制器' if controller_tag == 'PD' else 'ADRC控制器'
    t_hist = results['t']
    q_true_hist = results['q_true']
    q_est_hist = results['q_est']
    w_hist = results['w']
    u_hist = results['u']
    dist_true_hist = results['dist_true']
    dist_est_hist = results['dist_est']
    bias_true_hist = results['gyro_bias_true']
    bias_est_hist = results['gyro_bias_est']

    figures = {
        'attitude_error.png': plot_attitude_error(
            t_hist,
            q_true_hist,
            q_est_hist,
            f'{controller_name} | 姿态估计误差',
        ),
        'simulation_overview.png': plot_simulation_process_overview(
            results,
            title=f'{controller_name} | 闭环仿真过程总览',
        ),
        'simulation_report_dashboard.png': plot_simulation_report_dashboard(
            results,
            title=f"{controller_tag} 控制器 | 单次仿真综合仪表板",
        ),
    }

    if str(detail_level).lower() == 'full':
        figures.update({
            'attitude_estimation.png': plot_attitude_estimation(
                t_hist,
                q_true_hist,
                q_est_hist,
                f'{controller_name} | MEKF姿态估计结果',
            ),
            'control_response.png': plot_control_response(
                t_hist,
                results['err'],
                w_hist,
                u_hist,
                umax,
                f'{controller_name} | 控制系统响应',
            ),
            'angular_velocity.png': plot_angular_velocity(
                t_hist,
                w_hist,
                f'{controller_name} | 角速度响应',
            ),
            'control_torque.png': plot_control_torque(
                t_hist,
                u_hist,
                umax,
                f'{controller_name} | 控制力矩',
            ),
            'gyro_bias_estimation.png': plot_observer_tracking(
                t_hist,
                bias_true_hist,
                bias_est_hist,
                title=f'{controller_name} | 陀螺偏置估计对比',
                ylabel='偏置 (rad/s)',
                truth_label='真实偏置',
                estimate_label='MEKF估计偏置',
            ),
        })

    if controller_tag == 'ADRC':
        figures['disturbance_estimation.png'] = plot_observer_tracking(
            t_hist,
            dist_true_hist,
            dist_est_hist,
            title='ADRC 控制器 | 扰动估计跟踪',
            ylabel='力矩 (N·m)',
            truth_label='真实扰动',
            estimate_label='ESO估计扰动',
        )

    if results.get('inertia_estimator_scheme') is not None:
        figures['inertia_identification_dashboard.png'] = plot_inertia_identification_dashboard(
            results,
            title=f'{controller_name} | 动力学参数辨识综合图',
        )

    if str(detail_level).lower() == 'full' and results.get('inertia_estimator_scheme') is not None:
        figures['inertia_identification.png'] = plot_inertia_identification(
            results,
            title=f'{controller_name} | 在线惯量辨识过程',
        )

    return figures


def _simulate_pd_metrics_only(Kp, Kd, T, dt, umax, noise, seed, excitation_profile, controller_type, true_inertia=None):
    """
    面向 PD 自动调参的轻量仿真路径。

    该路径保持与完整仿真一致的初始激励与扰动序列，但跳过传感器/滤波器链路，
    仅计算目标函数所需的时域指标，显著降低优化器横评的单点评估成本。
    """
    rng = np.random.RandomState(seed)
    true_inertia_diag = np.diag(_resolve_true_inertia_matrix(true_inertia))
    nominal_inertia_diag = np.diag(np.asarray(CONFIG['spacecraft']['inertia'], dtype=float))
    rate_scale = np.sqrt(np.mean(true_inertia_diag) / max(np.mean(nominal_inertia_diag), 1e-12))
    q, w = _resolve_excitation_initial_state(rng, T, excitation_profile, rate_scale=rate_scale)
    steps = int(T / dt)

    # 保持随机数消费顺序与完整仿真一致，使扰动序列可复现实验结果。
    noise_batches = _prepare_random_batches(rng, steps, noise)
    dist_noise_samples = noise_batches['dist_noise_samples']

    dist_const = np.array([0.02, -0.015, 0.01], dtype=float)
    dist_noise_std = 0.003

    t_hist = np.empty(steps, dtype=float)
    t_hist[:] = np.arange(steps, dtype=float) * dt
    err_hist, u_norm_hist, sat_count = _simulate_pd_metrics_kernel(
        float(Kp),
        float(Kd),
        float(dt),
        float(umax),
        np.asarray(q, dtype=float),
        np.asarray(w, dtype=float),
        true_inertia_diag.copy(),
        dist_const,
        float(dist_noise_std),
        np.asarray(dist_noise_samples, dtype=float),
    )

    settle_time = _compute_settle_time(err_hist, t_hist)
    final_error = float(err_hist[-1])
    overshoot = _compute_overshoot(err_hist)
    effort = _curve_integral(u_norm_hist, t_hist)
    max_err = float(np.max(err_hist))

    return {
        'settle_time': settle_time,
        'overshoot': overshoot,
        'final_error': final_error,
        'max_err': max_err,
        'effort': effort,
        'IAE': _curve_integral(np.abs(err_hist), t_hist),
        'ITAE': _curve_integral(t_hist * np.abs(err_hist), t_hist),
        'ISU': _curve_integral(u_norm_hist ** 2, t_hist),
        'sat_ratio': sat_count / max(1, steps),
        'controller_type': controller_type,
        'control_feedback_source': 'truth',
        'u_limit': float(umax),
        'Kp': float(Kp),
        'Kd': float(Kd),
        'dist_const': dist_const.copy(),
        'inertia_estimator_scheme': None,
    }


def _save_simulation_data_tables(results, output_dir, controller_kind):
    """
    保存单次仿真的摘要指标与时序数据，便于后续论文制表或二次分析。
    """
    controller_tag = str(controller_kind).upper()
    metrics_rows = [
        {'metric': 'controller_type', 'value': results.get('controller_type', controller_tag), 'unit': '-'},
        {'metric': 'control_feedback_source', 'value': results.get('control_feedback_source', 'truth'), 'unit': '-'},
        {'metric': 'settle_time', 'value': f"{results['settle_time']:.6f}", 'unit': 's'},
        {'metric': 'overshoot', 'value': f"{results['overshoot']:.6f}", 'unit': 'deg'},
        {'metric': 'final_error', 'value': f"{results['final_error']:.6f}", 'unit': 'deg'},
        {'metric': 'peak_error', 'value': f"{results['peak_error']:.6f}", 'unit': 'deg'},
        {'metric': 'rms_error', 'value': f"{results['rms_error']:.6f}", 'unit': 'deg'},
        {'metric': 'steady_state_rms', 'value': f"{results['steady_state_rms']:.6f}", 'unit': 'deg'},
        {'metric': 'IAE', 'value': f"{results['IAE']:.6f}", 'unit': 'deg·s'},
        {'metric': 'ITAE', 'value': f"{results['ITAE']:.6f}", 'unit': 'deg·s²'},
        {'metric': 'ISU', 'value': f"{results['ISU']:.6f}", 'unit': 'N²·m²·s'},
        {'metric': 'effort', 'value': f"{results['effort']:.6f}", 'unit': 'N·m·s'},
        {'metric': 'peak_rate_deg', 'value': f"{results['peak_rate_deg']:.6f}", 'unit': 'deg/s'},
        {'metric': 'peak_torque', 'value': f"{results['peak_torque']:.6f}", 'unit': 'N·m'},
        {'metric': 'peak_axis_torque', 'value': f"{results['peak_axis_torque']:.6f}", 'unit': 'N·m'},
        {'metric': 'control_limit_norm', 'value': f"{results['control_limit_norm']:.6f}", 'unit': 'N·m'},
        {'metric': 'torque_usage_ratio', 'value': f"{results['torque_usage_ratio']:.6f}", 'unit': '-'},
        {'metric': 'sat_ratio', 'value': f"{results['sat_ratio']:.6f}", 'unit': '-'},
        {'metric': 'u_limit', 'value': f"{results['u_limit']:.6f}", 'unit': 'N·m'},
    ]
    if np.isfinite(results.get('attitude_est_rmse', np.nan)):
        metrics_rows.append({'metric': 'attitude_est_rmse', 'value': f"{results['attitude_est_rmse']:.6f}", 'unit': 'deg'})
    if np.isfinite(results.get('bias_rmse', np.nan)):
        metrics_rows.append({'metric': 'bias_rmse', 'value': f"{results['bias_rmse']:.6f}", 'unit': 'rad/s'})
    if np.isfinite(results.get('disturbance_rmse', np.nan)):
        metrics_rows.append({'metric': 'disturbance_rmse', 'value': f"{results['disturbance_rmse']:.6f}", 'unit': 'N·m'})
    if results.get('Kp') is not None:
        metrics_rows.append({'metric': 'Kp', 'value': f"{results['Kp']:.6f}", 'unit': '-'})
    if results.get('Kd') is not None:
        metrics_rows.append({'metric': 'Kd', 'value': f"{results['Kd']:.6f}", 'unit': '-'})
    if results.get('inertia_reference') is not None:
        inertia_ref = np.asarray(results['inertia_reference'], dtype=float)
        for idx, name in enumerate(['Jxx_ref', 'Jyy_ref', 'Jzz_ref']):
            metrics_rows.append({'metric': name, 'value': f"{inertia_ref[idx]:.6f}", 'unit': 'kg·m²'})
    _write_csv_rows(
        Path(output_dir) / 'summary_metrics.csv',
        ['metric', 'value', 'unit'],
        metrics_rows,
    )

    time_history_rows = []
    total_steps = len(results['t'])
    for idx in range(total_steps):
        time_history_rows.append({
            't_s': f"{results['t'][idx]:.6f}",
            'err_deg': f"{results['err'][idx]:.6f}",
            'q_true_w': f"{results['q_true'][idx, 0]:.8f}",
            'q_true_x': f"{results['q_true'][idx, 1]:.8f}",
            'q_true_y': f"{results['q_true'][idx, 2]:.8f}",
            'q_true_z': f"{results['q_true'][idx, 3]:.8f}",
            'q_est_w': f"{results['q_est'][idx, 0]:.8f}",
            'q_est_x': f"{results['q_est'][idx, 1]:.8f}",
            'q_est_y': f"{results['q_est'][idx, 2]:.8f}",
            'q_est_z': f"{results['q_est'][idx, 3]:.8f}",
            'wx_rad_s': f"{results['w'][idx, 0]:.8f}",
            'wy_rad_s': f"{results['w'][idx, 1]:.8f}",
            'wz_rad_s': f"{results['w'][idx, 2]:.8f}",
            'ux_Nm': f"{results['u'][idx, 0]:.8f}",
            'uy_Nm': f"{results['u'][idx, 1]:.8f}",
            'uz_Nm': f"{results['u'][idx, 2]:.8f}",
            'dist_true_x': f"{results['dist_true'][idx, 0]:.8f}",
            'dist_true_y': f"{results['dist_true'][idx, 1]:.8f}",
            'dist_true_z': f"{results['dist_true'][idx, 2]:.8f}",
            'dist_est_x': f"{results['dist_est'][idx, 0]:.8f}",
            'dist_est_y': f"{results['dist_est'][idx, 1]:.8f}",
            'dist_est_z': f"{results['dist_est'][idx, 2]:.8f}",
            'gyro_bias_true_x': f"{results['gyro_bias_true'][idx, 0]:.8f}",
            'gyro_bias_true_y': f"{results['gyro_bias_true'][idx, 1]:.8f}",
            'gyro_bias_true_z': f"{results['gyro_bias_true'][idx, 2]:.8f}",
            'gyro_bias_est_x': f"{results['gyro_bias_est'][idx, 0]:.8f}",
            'gyro_bias_est_y': f"{results['gyro_bias_est'][idx, 1]:.8f}",
            'gyro_bias_est_z': f"{results['gyro_bias_est'][idx, 2]:.8f}",
            'Jxx_est': f"{results['inertia_est'][idx, 0]:.8f}",
            'Jyy_est': f"{results['inertia_est'][idx, 1]:.8f}",
            'Jzz_est': f"{results['inertia_est'][idx, 2]:.8f}",
            'inertia_update_mask': f"{results['inertia_update_mask'][idx]:.0f}",
            'wdot_x': f"{results['wdot_est'][idx, 0]:.8f}",
            'wdot_y': f"{results['wdot_est'][idx, 1]:.8f}",
            'wdot_z': f"{results['wdot_est'][idx, 2]:.8f}",
            'regressor_min_sv': f"{results['regressor_min_sv'][idx]:.8f}",
            'regressor_jxx_norm': f"{results['axis_regressor_norms'][idx, 0]:.8f}",
            'regressor_jyy_norm': f"{results['axis_regressor_norms'][idx, 1]:.8f}",
            'regressor_jzz_norm': f"{results['axis_regressor_norms'][idx, 2]:.8f}",
        })
    _write_csv_rows(
        Path(output_dir) / 'time_history.csv',
        list(time_history_rows[0].keys()) if time_history_rows else ['t_s'],
        time_history_rows,
    )


def _save_simulation_plot_suite(results, output_dir, controller_kind, detail_level='core'):
    """
    保存单次仿真的完整图像集合与数据表。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = _build_simulation_figures(results, results['u_limit'], controller_kind, detail_level=detail_level)
    for filename, fig in figures.items():
        _save_figure(fig, output_dir / filename)
    _save_simulation_data_tables(results, output_dir, controller_kind)


def _run_inertia_identification_cases(run_profile, umax, output_layout, true_inertia):
    """
    运行两组动力学参数辨识案例，并保存专用图像与数据。
    """
    print(f"\n{'='*70}")
    print("运行动力学参数辨识案例...")
    print(f"{'='*70}")

    base_cfg = {
        'T': max(18.0, float(run_profile['T'])),
        'dt': float(run_profile['dt']),
        'umax': float(umax),
        'use_star_tracker': True,
        'show_plots': False,
        'seed': 7,
        'excitation_profile': 'aggressive',
        'true_inertia': ensure_diag_inertia(true_inertia),
    }
    case_cfg = _build_identification_case_configs()

    rls_results = simulate_attitude_control(
        controller_type='ADRC',
        adrc_params=_build_default_adrc_params(base_cfg['dt']),
        inertia_estimator_cfg=case_cfg['RLS'],
        **base_cfg,
    )
    _save_simulation_plot_suite(rls_results, output_layout['identification_rls'], 'ADRC', detail_level=OUTPUT_DETAIL_LEVEL)

    mekf_results = simulate_attitude_control(
        controller_type='ADRC',
        adrc_params=_build_default_adrc_params(base_cfg['dt']),
        inertia_estimator_cfg=case_cfg['MEKF'],
        seed=11,
        **{k: v for k, v in base_cfg.items() if k != 'seed'},
    )
    _save_simulation_plot_suite(mekf_results, output_layout['identification_mekf'], 'ADRC', detail_level=OUTPUT_DETAIL_LEVEL)

    print(f"RLS 辨识结果已保存到: {Path(output_layout['identification_rls']).resolve()}")
    print(f"MEKF 辨识结果已保存到: {Path(output_layout['identification_mekf']).resolve()}")
    return {
        'RLS': rls_results,
        'MEKF': mekf_results,
    }

def _plot_simulation_diagnostics(results, umax, controller_kind):
    """
    绘制单次仿真的诊断图。
    """
    _build_simulation_figures(results, umax, controller_kind)
    plt.show()


def _controller_comparison_metric_specs():
    """
    统一定义控制器对比指标的显示名称、单位与缩放方式。
    """
    return [
        {'key': 'settle_time', 'name': '调节时间', 'unit': 's', 'scale': 1.0, 'print_fmt': '.2f', 'table_fmt': '.4f'},
        {'key': 'overshoot', 'name': '超调量', 'unit': 'deg', 'scale': 1.0, 'print_fmt': '.2f', 'table_fmt': '.4f'},
        {'key': 'final_error', 'name': '稳态误差', 'unit': 'deg', 'scale': 1.0, 'print_fmt': '.3f', 'table_fmt': '.4f'},
        {'key': 'steady_state_rms', 'name': '尾段RMS误差', 'unit': 'deg', 'scale': 1.0, 'print_fmt': '.3f', 'table_fmt': '.4f'},
        {'key': 'IAE', 'name': 'IAE', 'unit': 'deg·s', 'scale': 1.0, 'print_fmt': '.3f', 'table_fmt': '.4f'},
        {'key': 'ITAE', 'name': 'ITAE', 'unit': 'deg·s^2', 'scale': 1.0, 'print_fmt': '.3f', 'table_fmt': '.4f'},
        {'key': 'ISU', 'name': 'ISU', 'unit': 'N^2·m^2·s', 'scale': 1.0, 'print_fmt': '.3f', 'table_fmt': '.4f'},
        {'key': 'effort', 'name': '控制能耗', 'unit': 'N·m·s', 'scale': 1.0, 'print_fmt': '.3f', 'table_fmt': '.4f'},
        {'key': 'peak_axis_torque', 'name': '峰值单轴力矩', 'unit': 'N·m', 'scale': 1.0, 'print_fmt': '.3f', 'table_fmt': '.4f'},
        {'key': 'sat_ratio', 'name': '饱和比例', 'unit': '%', 'scale': 100.0, 'print_fmt': '.1f', 'table_fmt': '.4f'},
    ]


def _safe_relative_improvement(baseline, candidate, zero_tol=1e-9):
    """
    计算相对改进百分比；当基准值接近零时返回 None，避免产生误导性超大百分比。
    """
    baseline = float(baseline)
    candidate = float(candidate)
    if abs(baseline) <= zero_tol:
        if abs(candidate - baseline) <= zero_tol:
            return 0.0
        return None
    return (baseline - candidate) / baseline * 100.0


def _format_improvement_pct(improvement, digits=1, na_text='N/A'):
    """
    统一格式化改进百分比字符串。
    """
    if improvement is None or not np.isfinite(improvement):
        return na_text
    return f"{improvement:+.{int(digits)}f}%"


def _print_controller_comparison(results_pd, results_adrc):
    """
    打印 PD 与 ADRC 的核心性能对比。
    """
    print(f"\n{'='*70}")
    print("性能对比结果")
    print(f"{'='*70}")
    print(f"\n{'指标':<20} {'PD控制器':<20} {'ADRC控制器':<20} {'改进百分比':<15}")
    print("-" * 75)

    for spec in _controller_comparison_metric_specs():
        pd_val = float(results_pd[spec['key']]) * spec['scale']
        adrc_val = float(results_adrc[spec['key']]) * spec['scale']
        improvement = _safe_relative_improvement(pd_val, adrc_val)
        print(
            f"{f'{spec['name']} ({spec['unit']})':<20} "
            f"{format(pd_val, spec['print_fmt']):<20} "
            f"{format(adrc_val, spec['print_fmt']):<20} "
            f"{_format_improvement_pct(improvement, digits=1):>13}"
        )

    print("\n[关键指标] Steady-state Error（稳态误差）")
    print(f"  PD   : {results_pd['final_error']:.4f} deg")
    print(f"  ADRC : {results_adrc['final_error']:.4f} deg")
    if results_adrc['final_error'] < results_pd['final_error']:
        print("  结论 : ADRC 在常值干扰下静差更小（符合抗扰预期）。")
    else:
        print("  结论 : 当前参数下 ADRC 静差未优于 PD，建议继续提升 omega_o 或延长仿真时间。")


def _save_controller_comparison_summary(results_pd, results_adrc, output_dir):
    """
    保存 PD 与 ADRC 的对比指标表，便于直接插入报告。
    """
    rows = []
    for spec in _controller_comparison_metric_specs():
        pd_raw = float(results_pd[spec['key']])
        adrc_raw = float(results_adrc[spec['key']])
        pd_val = pd_raw * spec['scale']
        adrc_val = adrc_raw * spec['scale']
        improvement = _safe_relative_improvement(pd_val, adrc_val)
        rows.append({
            'metric_key': spec['key'],
            'metric_name': spec['name'],
            'unit': spec['unit'],
            'pd_value': f"{pd_val:.6f}",
            'adrc_value': f"{adrc_val:.6f}",
            'improvement_pct': '' if improvement is None else f"{improvement:.3f}",
        })
    _write_csv_rows(
        Path(output_dir) / 'controller_comparison_summary.csv',
        ['metric_key', 'metric_name', 'unit', 'pd_value', 'adrc_value', 'improvement_pct'],
        rows,
    )


def _remove_comparison_plot_files(output_dir):
    """
    在快速模式下删除旧对比图，避免报告引用历史运行残留文件。
    """
    output_dir = Path(output_dir)
    for filename in (
        'pd_vs_adrc_comparison.png',
        'pd_vs_adrc_time_response_overlay.png',
        'adrc_disturbance_estimation.png',
    ):
        path = output_dir / filename
        if path.exists():
            path.unlink()


def _build_controller_comparison_rows(results_pd, results_adrc):
    """
    统一生成控制器对比表的 Markdown 行数据。
    """
    rows = []
    for spec in _controller_comparison_metric_specs():
        pd_val = float(results_pd[spec['key']]) * spec['scale']
        adrc_val = float(results_adrc[spec['key']]) * spec['scale']
        improvement = _safe_relative_improvement(pd_val, adrc_val)
        rows.append([
            spec['name'],
            format(pd_val, spec['table_fmt']),
            format(adrc_val, spec['table_fmt']),
            spec['unit'],
            _format_improvement_pct(improvement, digits=2),
        ])
    return rows


def _build_optimizer_ranking_rows(tuning):
    """
    统一生成优化器排序表的 Markdown 行数据。
    """
    rows = []
    for idx, row in enumerate(tuning.get('ranking', []), start=1):
        rows.append([
            idx,
            row['method'],
            f"{row['Kp']:.4f}",
            f"{row['Kd']:.4f}",
            f"{row['score']:.4f}",
            f"{row['settle_time']:.4f}",
            f"{row['final_error']:.4f}",
            f"{row['effort']:.4f}",
        ])
    return rows


def _format_best_optimizer_summary(tuning):
    """
    生成最优优化器的一行摘要文本。
    """
    best_method = tuning.get('best')
    if best_method is None:
        return "无"
    return (
        f"{best_method['method']} (Kp={best_method['Kp']:.4f}, "
        f"Kd={best_method['Kd']:.4f}, score={best_method['score']:.4f})"
    )


def _format_selected_inertia_report(selected_inertia_summary):
    """
    生成前置辨识选型的摘要与验证说明文本。
    """
    if selected_inertia_summary is None:
        return "未使用前置辨识结果", ""

    selected_inertia = ensure_diag_inertia(selected_inertia_summary.get('inertia', [0.50, 0.50, 0.50]))
    selected_scheme = selected_inertia_summary.get('scheme', 'fallback')
    selected_update_ratio = float(selected_inertia_summary.get('update_ratio', 0.0))
    selected_inertia_text = (
        f"{selected_scheme}: "
        f"[{selected_inertia[0]:.4f}, {selected_inertia[1]:.4f}, {selected_inertia[2]:.4f}] kg·m² "
        f"(更新占比 {selected_update_ratio:.1f}%)"
    )

    validation_score = selected_inertia_summary.get('validation_score')
    validation_settle = selected_inertia_summary.get('validation_settle_time')
    validation_final_error = selected_inertia_summary.get('validation_final_error')
    selected_inertia_detail = ""
    if validation_score is not None and np.isfinite(validation_score):
        selected_inertia_detail = (
            f"候选验证得分={validation_score:.4f}, "
            f"Ts={float(validation_settle):.3f}s, "
            f"Final={float(validation_final_error):.4f}deg"
        )
    return selected_inertia_text, selected_inertia_detail


def _format_adrc_tuning_report(adrc_tuning):
    """
    生成 ADRC 自动整定摘要文本。
    """
    if adrc_tuning is None or adrc_tuning.get('best') is None:
        return "未执行 ADRC 自动整定"

    best_adrc = adrc_tuning['best']
    return (
        f"omega_c={best_adrc['omega_c']:.4f}, omega_o={best_adrc['omega_o']:.4f}, "
        f"ratio={best_adrc['omega_ratio']:.3f}, score={best_adrc['score']:.4f}"
    )


def _resolve_controller_tradeoff_text(results_pd, results_adrc):
    """
    生成控制器对比的文字解读。
    """
    effort_delta = float(results_adrc['effort'] - results_pd['effort'])
    if results_adrc['final_error'] < results_pd['final_error'] and effort_delta > 0.0:
        return "ADRC 显著降低了稳态误差和调节时间，但为此付出了更高的控制能耗与单轴峰值力矩。"
    if results_adrc['final_error'] < results_pd['final_error']:
        return "ADRC 在精度和速度上同时优于 PD，且没有增加明显的控制开销。"
    return "当前参数下 ADRC 的精度收益不明显，建议继续检查带宽整定与扰动补偿配置。"


def _build_report_image_blocks(output_layout, run_profile, identification_results):
    """
    统一生成报告中要插入的图像块列表。
    """
    image_blocks = [
        ("优化综合仪表板", output_layout['optimization'] / 'pd_optimizer_report_dashboard.png'),
        ("优化景观图", output_layout['optimization'] / 'pd_optimizer_landscape.png'),
        ("优化器收敛统计图", output_layout['optimization'] / 'pd_optimizer_convergence_statistics.png'),
        ("优化器最优闭环响应对比图", output_layout['optimization'] / 'pd_optimizer_best_response_dashboard.png'),
        ("优化器性能权衡图", output_layout['optimization'] / 'pd_optimizer_tradeoff_scatter.png'),
        ("优化器指标热力图", output_layout['optimization'] / 'pd_optimizer_metric_heatmap.png'),
        ("PD 单次仿真综合仪表板", output_layout['simulation_pd'] / 'simulation_report_dashboard.png'),
        ("ADRC 单次仿真综合仪表板", output_layout['simulation_adrc'] / 'simulation_report_dashboard.png'),
    ]
    if run_profile.get('save_comparison_plots', False):
        image_blocks.extend([
            ("控制器性能对比图", output_layout['comparison'] / 'pd_vs_adrc_comparison.png'),
            ("控制器时域叠加图", output_layout['comparison'] / 'pd_vs_adrc_time_response_overlay.png'),
        ])
    if identification_results:
        image_blocks.extend([
            ("RLS 参数辨识综合图", output_layout['identification_rls'] / 'inertia_identification_dashboard.png'),
            ("MEKF 参数辨识综合图", output_layout['identification_mekf'] / 'inertia_identification_dashboard.png'),
        ])
    return image_blocks


def _build_identification_summary_rows(identification_results):
    """
    统一生成参数辨识摘要表数据。
    """
    rows = []
    for scheme_name, res in identification_results.items():
        inertia_ref = np.asarray(res.get('inertia_reference', res['inertia_est'][0]), dtype=float)
        inertia_final = np.asarray(res['inertia_est'][-1], dtype=float)
        err_norm = float(np.linalg.norm(inertia_final - inertia_ref))
        update_ratio = float(np.mean(np.asarray(res.get('inertia_update_mask', []), dtype=float) > 0.5) * 100.0)
        rows.append([
            scheme_name,
            f"[{inertia_ref[0]:.4f}, {inertia_ref[1]:.4f}, {inertia_ref[2]:.4f}]",
            f"[{inertia_final[0]:.4f}, {inertia_final[1]:.4f}, {inertia_final[2]:.4f}]",
            f"{err_norm:.4e}",
            f"{update_ratio:.1f}%",
        ])
    return rows


def _build_report_file_index_rows(root_dir, output_layout, adrc_tuning, identification_results):
    """
    统一生成报告中文件索引表数据。
    """
    rows = [
        ['优化结果', _relative_markdown_path(root_dir, output_layout['optimization']), '优化图像、已评估点与排序表'],
        ['PD 仿真', _relative_markdown_path(root_dir, output_layout['simulation_pd']), 'PD 单次仿真图像与时序数据'],
        ['ADRC 仿真', _relative_markdown_path(root_dir, output_layout['simulation_adrc']), 'ADRC 单次仿真图像与时序数据'],
        ['对比结果', _relative_markdown_path(root_dir, output_layout['comparison']), 'PD vs ADRC 对比图和摘要表'],
    ]
    if adrc_tuning is not None and adrc_tuning.get('output_dir'):
        rows.append([
            'ADRC 整定',
            _relative_markdown_path(root_dir, adrc_tuning['output_dir']),
            'ADRC 带宽整定结果与排序表',
        ])
    if identification_results:
        rows.extend([
            ['RLS 辨识', _relative_markdown_path(root_dir, output_layout['identification_rls']), 'RLS 参数辨识图像与时序数据'],
            ['MEKF 辨识', _relative_markdown_path(root_dir, output_layout['identification_mekf']), '增广 MEKF 参数辨识图像与时序数据'],
        ])
    return rows


def _generate_markdown_report(
    tuning,
    results_pd,
    results_adrc,
    output_layout,
    run_profile,
    identification_results=None,
    selected_inertia_summary=None,
    adrc_tuning=None,
):
    """
    自动生成 Markdown 风格结果报告，串联关键图表和指标。
    """
    root_dir = Path(output_layout['root'])
    report_path = root_dir / 'simulation_summary_report.md'
    generated_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    comparison_rows = _build_controller_comparison_rows(results_pd, results_adrc)
    ranking_rows = _build_optimizer_ranking_rows(tuning)
    best_summary = _format_best_optimizer_summary(tuning)
    selected_inertia_text, selected_inertia_detail = _format_selected_inertia_report(selected_inertia_summary)
    adrc_tuning_text = _format_adrc_tuning_report(adrc_tuning)
    tradeoff_text = _resolve_controller_tradeoff_text(results_pd, results_adrc)
    image_blocks = _build_report_image_blocks(output_layout, run_profile, identification_results)

    lines = [
        "# 卫星姿态控制仿真结果汇总报告",
        "",
        f"- 生成时间: `{generated_time}`",
        f"- 运行档位: `{run_profile['profile']}`",
        f"- 仿真时长: `{run_profile['T']}` s",
        f"- 时间步长: `{run_profile['dt']}` s",
        f"- PD 闭环反馈源: `{results_pd.get('control_feedback_source', 'truth')}`",
        f"- ADRC 闭环反馈源: `{results_adrc.get('control_feedback_source', 'truth')}`",
        f"- 前置辨识采用惯量: `{selected_inertia_text}`",
        f"- ADRC 整定结果: `{adrc_tuning_text}`",
        f"- 优化最优方法: `{best_summary}`",
        "",
        "## 1. 结论摘要",
        "",
        "- 本次流程采用“先辨识，后调参，再正式控制对比”的实验顺序。",
        f"- PD 控制器末端姿态误差: `{results_pd['final_error']:.4f} deg`",
        f"- ADRC 控制器末端姿态误差: `{results_adrc['final_error']:.4f} deg`",
        f"- PD 控制器调节时间: `{results_pd['settle_time']:.4f} s`",
        f"- ADRC 控制器调节时间: `{results_adrc['settle_time']:.4f} s`",
        f"- PD 尾段 RMS 误差: `{results_pd['steady_state_rms']:.4f} deg`",
        f"- ADRC 尾段 RMS 误差: `{results_adrc['steady_state_rms']:.4f} deg`",
        f"- PD 控制器控制能耗: `{results_pd['effort']:.4f} N·m·s`",
        f"- ADRC 控制器控制能耗: `{results_adrc['effort']:.4f} N·m·s`",
        f"- 峰值单轴力矩对比: `PD={results_pd['peak_axis_torque']:.4f} N·m, ADRC={results_adrc['peak_axis_torque']:.4f} N·m`",
        f"- 结果解读: `{tradeoff_text}`",
        "",
        "## 2. 控制器性能对比表",
        "",
        _markdown_table(
            ['指标', 'PD', 'ADRC', '单位', 'ADRC 相对 PD 改进'],
            comparison_rows,
        ),
        "",
        "## 3. 优化器排序结果",
        "",
    ]

    if ranking_rows:
        lines.extend([
            _markdown_table(
                ['排名', '算法', 'Kp', 'Kd', 'Score', 'Ts(s)', 'Final Error(deg)', 'Effort(N·m·s)'],
                ranking_rows,
            ),
            "",
        ])
    else:
        lines.extend([
            "本次运行未生成优化器排序结果。",
            "",
        ])

    lines.extend([
        "## 4. 关键图像",
        "",
    ])

    if not run_profile.get('save_comparison_plots', False):
        lines.extend([
            "- 当前为快速模式，本次运行未刷新控制器对比图，仅保留本轮生成的单控制器图像与摘要表。",
            "",
        ])

    for title, img_path in image_blocks:
        if Path(img_path).exists():
            rel_path = _relative_markdown_path(root_dir, img_path)
            lines.extend([
                f"### {title}",
                "",
                f"![{title}]({rel_path})",
                "",
            ])

    lines.extend([
        "## 5. 动力学参数辨识摘要",
        "",
    ])

    if identification_results:
        id_rows = _build_identification_summary_rows(identification_results)
        lines.extend([
            _markdown_table(
                ['方案', '参考惯量', '最终估计', '最终误差范数', '更新占比'],
                id_rows,
            ),
            "",
            f"- 后续调参与 ADRC 参数配置采用: `{selected_inertia_text}`",
            f"- 选中模型的短时闭环验证: `{selected_inertia_detail or '未提供验证摘要'}`",
            "",
        ])
    else:
        lines.extend([
            "本次运行未生成参数辨识案例。",
            "",
        ])

    lines.extend([
        "## 6. ADRC 整定摘要",
        "",
    ])

    if adrc_tuning is not None and adrc_tuning.get('best') is not None:
        best_adrc = adrc_tuning['best']
        lines.extend([
            f"- 最优带宽: `omega_c={best_adrc['omega_c']:.4f}, omega_o={best_adrc['omega_o']:.4f}`",
            f"- 整定目标值: `{best_adrc['score']:.4f}`",
            f"- 整定阶段指标: `Ts={best_adrc['settle_time']:.4f}s, Final={best_adrc['final_error']:.4f}deg, Overshoot={best_adrc['overshoot']:.4f}deg`",
            "",
        ])
    else:
        lines.extend([
            "本次运行未生成 ADRC 自动整定结果。",
            "",
        ])

    lines.extend([
        "## 7. 结果文件索引",
        "",
    ])

    file_index_rows = _build_report_file_index_rows(
        root_dir=root_dir,
        output_layout=output_layout,
        adrc_tuning=adrc_tuning,
        identification_results=identification_results,
    )

    lines.extend([
        _markdown_table(
            ['类别', '目录', '说明'],
            file_index_rows,
        ),
        "",
        "## 8. 附注",
        "",
        "- 报告中的图片路径均为相对于本 Markdown 文件的本地路径。",
        "- 若需继续写论文，可直接基于本报告提炼“实验设置 / 结果分析 / 图表说明”三个章节。",
        "",
    ])

    report_path.write_text("\n".join(lines), encoding='utf-8')
    return report_path


def _save_controller_comparison_plots(results_pd, results_adrc, umax, output_dir):
    """
    保存 PD 与 ADRC 的核心对比图。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n生成并保存对比图表...")
    fig = plot_controller_comparison_dashboard(results_pd, results_adrc, umax, title='PD控制器 vs ADRC控制器性能对比')
    _save_figure(fig, output_dir / "pd_vs_adrc_comparison.png")

    fig_overlay = plot_multiple_responses(
        {
            'PD': results_pd,
            'ADRC': results_adrc,
        },
        title='PD 与 ADRC 时域响应叠加对比',
    )
    _save_figure(fig_overlay, output_dir / "pd_vs_adrc_time_response_overlay.png")

    fig_dist = plot_observer_tracking(
        results_adrc['t'],
        results_adrc['dist_true'],
        results_adrc['dist_est'],
        title='ADRC 扰动估计对比（真实干扰 vs ESO估计）',
        ylabel='力矩 (N·m)',
        truth_label='真实扰动',
        estimate_label='ESO估计扰动',
    )
    _save_figure(fig_dist, output_dir / "adrc_disturbance_estimation.png")
    _save_controller_comparison_summary(results_pd, results_adrc, output_dir)
    print(f"图像已保存到: {output_dir.resolve()}")


def simulate_attitude_control(Kp=1.0, Kd=0.1, 
                               T=CONFIG['simulation']['T'], 
                               dt=CONFIG['simulation']['dt'], 
                               umax=CONFIG['spacecraft']['u_max'], 
                               fov_deg=CONFIG['sensors']['star_tracker']['fov'], 
                               noise=CONFIG['sensors']['star_tracker']['noise_std'], 
                               seed=CONFIG['simulation']['seed'], 
                               use_star_tracker=True, show_plots=True,
                               controller_type='PD', adrc_params=None,
                               inertia_estimator_cfg=None,
                               excitation_profile='auto',
                               true_inertia=None,
                               control_feedback_source='truth',
                               result_mode='full',
                               verbose=True):
    """
    完整的卫星姿态控制仿真
    
    参数:
        Kp: 比例增益（PD控制器）
        Kd: 微分增益（PD控制器）
        T: 仿真时间 (s)
        dt: 时间步长 (s)
        umax: 最大控制力矩 (N·m)
        fov_deg: 星敏感器视场角 (度)
        noise: 星敏感器噪声标准差
        seed: 随机种子
        use_star_tracker: 是否使用星敏感器（False时使用理想测量）
        show_plots: 是否显示图形
        controller_type: 控制器类型 ('PD' 或 'ADRC')
        adrc_params: ADRC控制器参数字典（可选）
        inertia_estimator_cfg: 在线惯量辨识配置
        excitation_profile: 激励档位 ('auto'/'nominal'/'aggressive')
        true_inertia: 真实转动惯量，对角元向量或 3x3 矩阵
        control_feedback_source: 控制闭环反馈源 ('truth'/'estimate')
        result_mode: 返回模式 ('full'/'metrics')；PD 调参时可走轻量指标路径
    
    返回:
        results: 包含仿真结果的字典
    """
    # 固定随机种子，保证 PD 与 ADRC 在各自仿真中经历完全一致的随机序列
    result_mode = str(result_mode).strip().lower()
    controller_kind = controller_type.upper()
    control_feedback_source = str(control_feedback_source).strip().lower()
    if (
        result_mode == 'metrics'
        and controller_kind == 'PD'
        and not use_star_tracker
        and inertia_estimator_cfg is None
        and not show_plots
        and control_feedback_source == 'truth'
    ):
        return _simulate_pd_metrics_only(
            Kp=Kp,
            Kd=Kd,
            T=T,
            dt=dt,
            umax=umax,
            noise=noise,
            seed=seed,
            excitation_profile=excitation_profile,
            controller_type=controller_type,
            true_inertia=true_inertia,
        )
    if control_feedback_source not in {'truth', 'estimate'}:
        raise ValueError("control_feedback_source must be 'truth' or 'estimate'")

    rng = np.random.RandomState(seed)
    
    # 初始化系统
    sc = Spacecraft(J=_resolve_true_inertia_matrix(true_inertia), umax=umax)
    st = (
        StarTracker(
            n_stars_catalog=1600,
            fov_deg=fov_deg,
            dir_noise_std=noise,
            rng=rng,
        )
        if use_star_tracker else None
    )
    
    adrc_controller, adrc_params = _init_controller(
        controller_kind=controller_kind,
        dt=dt,
        Kp=Kp,
        Kd=Kd,
        adrc_params=adrc_params,
        verbose=verbose,
    )
    
    # 初始条件
    true_inertia_diag = np.diag(_resolve_true_inertia_matrix(true_inertia))
    nominal_inertia_diag = np.diag(np.asarray(CONFIG['spacecraft']['inertia'], dtype=float))
    rate_scale = np.sqrt(np.mean(true_inertia_diag) / max(np.mean(nominal_inertia_diag), 1e-12))
    q, w = _resolve_excitation_initial_state(rng, T, excitation_profile, rate_scale=rate_scale)
    qd = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)  # 期望姿态（单位四元数）
    
    estimation_ctx = _build_estimation_context(
        sc=sc,
        inertia_estimator_cfg=inertia_estimator_cfg,
        initial_q=q,
        dt=dt,
    )
    mekf = estimation_ctx['mekf']
    use_augmented_mekf = estimation_ctx['use_augmented_mekf']
    inertia_scheme = estimation_ctx['inertia_scheme']
    inertia_identifier = estimation_ctx['inertia_identifier']
    inertia_reference = estimation_ctx['inertia_reference']
    estimator_cfg = estimation_ctx['estimator_cfg']
    inertia_warmup_steps = estimation_ctx['inertia_warmup_steps']
    wdot_filter_alpha = estimation_ctx['wdot_filter_alpha']
    wdot_regression_window = estimation_ctx['wdot_regression_window']
    disturbance_filter_alpha = estimation_ctx['disturbance_filter_alpha']
    disturbance_enable_steps = estimation_ctx['disturbance_enable_steps']
    disturbance_clip = estimation_ctx['disturbance_clip']
    regressor_min_sv_reference = estimation_ctx['regressor_min_sv_reference']
    
    steps = int(T / dt)
    history = _allocate_simulation_history(steps)
    sat_count = 0
    u_prev = np.zeros(3)  # 用于ADRC的上一时刻控制量
    w_filt_window = deque(maxlen=wdot_regression_window)
    wdot_filt_prev = None
    dist_torque_filt_prev = None
    nominal_J_diag = estimation_ctx['nominal_J_diag']
    current_J_diag = estimation_ctx['current_J_diag']

    # 外部扰动：常值项 + 小幅随机噪声
    # 常值项用于体现 ADRC 对偏置扰动（如重力梯度/气动不平衡）的抑制能力
    dist_const = np.array([0.02, -0.015, 0.01], dtype=float)
    dist_noise_std = 0.003
    # 陀螺仪真实偏置模型：常值初始偏置 + 随机游走
    true_bias = np.array([0.01, -0.015, 0.005], dtype=float)
    gyro_bias_rw_std = 1e-5
    gyro_noise_std = 0.002

    # 批量预生成循环内会反复用到的随机量，减少 Python 层频繁调用开销。
    noise_batches = _prepare_random_batches(rng, steps, noise)
    fallback_noise_axes = noise_batches['fallback_noise_axes']
    fallback_startracker_angles = noise_batches['fallback_startracker_angles']
    fallback_ideal_angles = noise_batches['fallback_ideal_angles']
    gyro_bias_rw_noise = noise_batches['gyro_bias_rw_noise']
    gyro_measurement_noise = noise_batches['gyro_measurement_noise']
    dist_noise_samples = noise_batches['dist_noise_samples']
    
    # 主仿真循环
    for k in range(steps):
        t = k * dt
        q_meas = _resolve_attitude_measurement(
            q=q,
            use_star_tracker=use_star_tracker,
            star_tracker=st,
            step_idx=k,
            fallback_noise_axes=fallback_noise_axes,
            fallback_startracker_angles=fallback_startracker_angles,
            fallback_ideal_angles=fallback_ideal_angles,
        )
        
        # 真实陀螺偏置随机游走，并生成含偏置测量值
        true_bias += gyro_bias_rw_std * gyro_bias_rw_noise[k]
        w_gyro = w + true_bias + gyro_noise_std * gyro_measurement_noise[k]
        if use_augmented_mekf:
            mekf.predict(w_gyro, dt, u_applied=u_prev)
        else:
            mekf.predict(w_gyro, dt)
        
        # MEKF更新
        mekf.update(q_meas)

        # 近似角加速度：对最近若干拍滤波角速度做短窗回归，降低差分放噪问题
        w_filt = w_gyro - mekf.b
        w_filt_window.append(w_filt.copy())
        omega_id, wdot_raw = _estimate_angular_rate_regression_state(w_filt_window, dt)
        wdot_est = _blend_vector(wdot_filt_prev, wdot_raw, wdot_filter_alpha)
        wdot_filt_prev = wdot_est.copy()
        q_ctrl, w_ctrl, _ = _resolve_control_feedback_state(
            control_feedback_source=control_feedback_source,
            q_true=q,
            w_true=w,
            q_est=mekf.q,
            w_est=w_filt,
        )

        dist_torque_est, dist_torque_filt_prev = _resolve_disturbance_torque_estimate(
            step_idx=k,
            inertia_scheme=inertia_scheme,
            inertia_identifier=inertia_identifier,
            adrc_controller=adrc_controller,
            disturbance_filter_alpha=disturbance_filter_alpha,
            disturbance_clip=disturbance_clip,
            disturbance_enable_steps=disturbance_enable_steps,
            dist_torque_filt_prev=dist_torque_filt_prev,
        )
        current_J_diag, inertia_updated, regressor_min_sv, axis_regressor_norms = _update_inertia_estimation(
            step_idx=k,
            inertia_scheme=inertia_scheme,
            inertia_identifier=inertia_identifier,
            use_augmented_mekf=use_augmented_mekf,
            mekf=mekf,
            inertia_warmup_steps=inertia_warmup_steps,
            omega_id=omega_id,
            wdot_est=wdot_est,
            u_prev=u_prev,
            dist_torque_est=dist_torque_est,
            nominal_J_diag=nominal_J_diag,
            adrc_controller=adrc_controller,
        )
        u, dist_est = _compute_control_command(
            controller_kind=controller_kind,
            qd=qd,
            q=q_ctrl,
            w=w_ctrl,
            Kp=Kp,
            Kd=Kd,
            adrc_controller=adrc_controller,
            dt=dt,
            u_prev=u_prev,
        )

        if inertia_identifier is not None:
            u = u + _compute_identification_probe_torque(t, estimator_cfg, umax)
        
        # 外部扰动：常值 + 随机白噪声
        dist = dist_const + dist_noise_std * dist_noise_samples[k]
        u_sat = np.clip(u, -sc.umax, sc.umax)
        q0_abs = min(1.0, max(0.0, float(abs(q[0]))))
        err_deg = np.rad2deg(2.0 * np.arccos(q0_abs))

        _record_simulation_sample(
            history=history,
            step_idx=k,
            t=t,
            q=q,
            q_est=mekf.q,
            w=w,
            u_sat=u_sat,
            err_deg=err_deg,
            dist_true=dist,
            dist_est=dist_est,
            gyro_bias_true=true_bias,
            gyro_bias_est=mekf.b,
            current_J_diag=current_J_diag,
            inertia_updated=inertia_updated,
            wdot_est=wdot_est,
            regressor_min_sv=regressor_min_sv,
            axis_regressor_norms=axis_regressor_norms,
        )
        
        # 统计饱和次数（按任一轴达到限幅计）。
        if np.any(np.abs(u_sat) >= (umax - 1e-6)):
            sat_count += 1
        
        # 动力学积分
        q, w, _ = sc.step(q, w, u, dt, dist=dist)
        
        # 更新上一时刻控制量（用于ADRC）
        u_prev = u_sat.copy()
    
    # 改进性能指标计算
    # 修复: settle_time判断逻辑 - 多级阈值法，提高算法间的性能差异
    settle_time = _compute_settle_time(history['err'], history['t'])
    results = _assemble_simulation_results(
        history=history,
        dist_const=dist_const,
        settle_time=settle_time,
        sat_count=sat_count,
        umax=umax,
        controller_kind=controller_kind,
        controller_type=controller_type,
        Kp=Kp,
        Kd=Kd,
        inertia_scheme=inertia_scheme,
        control_feedback_source=control_feedback_source,
        inertia_reference=inertia_reference,
        regressor_min_sv_reference=regressor_min_sv_reference,
    )
    
    # 绘图
    if show_plots:
        _plot_simulation_diagnostics(results, umax, controller_kind)
    
    return results


def _score_pd_results(results, weights):
    """将PD仿真结果映射到单一优化目标（越小越好）。
    
    改进:
    1) 对各指标进行非线性压缩，提升可分辨性
    2) 对失稳工况施加巨额惩罚，避免目标函数平顶
    """
    # 获取各项指标并进行标准化处理
    settle_time = float(results['settle_time'])
    overshoot = float(results['overshoot'])
    final_error = float(results['final_error'])
    effort = float(results['effort'])
    sat_ratio = float(results['sat_ratio'])
    max_err = float(results.get('max_err', np.max(np.asarray(results.get('err', [final_error]), dtype=float))))
    
    # 使用 sqrt / log 压缩，减小极值对搜索方向的破坏
    settle_term = weights['settle_time'] * np.sqrt(settle_time)  # sqrt压缩快速变化
    overshoot_term = weights['overshoot'] * (overshoot ** 0.8)  # 非线性压缩
    error_term = weights['final_error'] * np.log1p(final_error * 10)  # log压缩
    effort_term = weights['effort'] * np.sqrt(effort)
    sat_term = weights['sat_ratio'] * sat_ratio * 100  # 放大饱和率的影响
    
    # 失稳惩罚：大误差或接近翻转时直接抬高目标值
    instability_penalty = 0.0
    if final_error > 10.0 or max_err > 180.0:
        instability_penalty = 1000.0 + max_err * 10.0

    # 计算总目标函数
    score = settle_term + overshoot_term + error_term + effort_term + sat_term + instability_penalty
    
    return float(score)


def _score_adrc_results(results, weights=None):
    """
    将 ADRC 仿真结果映射到单一优化目标（越小越好）。

    相比 PD 评分，更强调 ITAE / ISU / 调节时间等学术常用指标，
    同时保留对超调、稳态误差和饱和的约束。
    """
    if weights is None:
        weights = {
            'settle_time': 2.8,
            'overshoot': 0.45,
            'final_error': 1.2,
            'IAE': 0.15,
            'ITAE': 2.10,
            'ISU': 0.10,
            'sat_ratio': 1.4,
        }

    settle_time = float(results['settle_time'])
    overshoot = float(results['overshoot'])
    final_error = float(results['final_error'])
    iae = float(results.get('IAE', results.get('final_error', 0.0)))
    itae = float(results.get('ITAE', iae))
    isu = float(results.get('ISU', results.get('effort', 0.0)))
    sat_ratio = float(results['sat_ratio'])
    max_err = float(np.max(np.asarray(results.get('err', [final_error]), dtype=float)))

    settle_term = weights['settle_time'] * np.sqrt(max(settle_time, 0.0))
    overshoot_term = weights['overshoot'] * (max(overshoot, 0.0) ** 0.72)
    error_term = weights['final_error'] * np.log1p(max(final_error, 0.0) * 30.0)
    iae_term = weights['IAE'] * np.log1p(max(iae, 0.0))
    itae_term = weights['ITAE'] * np.log1p(max(itae, 0.0))
    isu_term = weights['ISU'] * np.log1p(max(isu, 0.0) * 35.0)
    sat_term = weights['sat_ratio'] * sat_ratio * 100.0

    instability_penalty = 0.0
    if final_error > 3.0 or max_err > 120.0:
        instability_penalty = 300.0 + 2.0 * max_err

    return float(
        settle_term
        + overshoot_term
        + error_term
        + iae_term
        + itae_term
        + isu_term
        + sat_term
        + instability_penalty
    )


def _summarize_identified_inertia_result(result, smoothing_ratio=0.12, min_window=12):
    """
    对单条辨识结果做尾段平滑，提取可用于后续调参与控制配置的名义惯量。
    """
    inertia_hist = np.asarray(result.get('inertia_est'), dtype=float)
    if inertia_hist.ndim != 2 or inertia_hist.shape[0] == 0:
        return None

    window = int(max(min_window, round(inertia_hist.shape[0] * float(smoothing_ratio))))
    window = min(window, inertia_hist.shape[0])
    tail = inertia_hist[-window:]
    final_inertia = ensure_diag_inertia(np.mean(tail, axis=0))
    stability_norm = float(np.linalg.norm(np.std(tail, axis=0)))

    update_mask = np.asarray(result.get('inertia_update_mask', []), dtype=float)
    update_ratio = float(np.mean(update_mask > 0.5) * 100.0) if update_mask.size else 0.0
    axis_regressor_norms = np.asarray(result.get('axis_regressor_norms', np.full((inertia_hist.shape[0], 3), np.nan)), dtype=float)
    finite_axis_mask = np.all(np.isfinite(axis_regressor_norms), axis=1)
    if axis_regressor_norms.ndim == 2 and axis_regressor_norms.shape[1] == 3 and np.any(finite_axis_mask):
        mean_axis_regressor_norms = np.mean(axis_regressor_norms[finite_axis_mask], axis=0)
        max_axis_norm = float(np.max(mean_axis_regressor_norms))
        excitation_balance = (
            float(np.min(mean_axis_regressor_norms) / max(max_axis_norm, 1e-12))
            if max_axis_norm > 0.0 else 0.0
        )
    else:
        mean_axis_regressor_norms = np.full(3, np.nan, dtype=float)
        excitation_balance = np.nan

    summary = {
        'inertia': final_inertia,
        'stability_norm': stability_norm,
        'update_ratio': update_ratio,
        'tail_window': int(window),
        'mean_axis_regressor_norms': mean_axis_regressor_norms,
        'excitation_balance': excitation_balance,
    }

    if result.get('inertia_reference') is not None:
        inertia_ref = ensure_diag_inertia(result['inertia_reference'])
        summary['reference_error_norm'] = float(np.linalg.norm(final_inertia - inertia_ref))

    return summary


def _resolve_validation_seeds(validation_seed, validation_runs):
    """
    将惯量候选验证的随机种子统一展开为整数列表。
    """
    runs = max(1, int(validation_runs))
    if np.isscalar(validation_seed):
        seed0 = int(validation_seed)
        return [seed0 + i for i in range(runs)]

    seeds = [int(seed) for seed in validation_seed]
    if not seeds:
        return [0]
    return seeds[:runs]


def _build_identification_selection_candidates(identification_results, fallback):
    """
    构造用于后续控制配置选择的惯量候选集合。
    """
    candidates = []
    for scheme_name, result in identification_results.items():
        summary = _summarize_identified_inertia_result(result)
        if summary is None:
            continue
        summary = dict(summary)
        summary.update({
            'scheme': str(scheme_name).upper(),
            'selection_origin': 'identified',
        })
        candidates.append(summary)

    if len(candidates) >= 2:
        fused_inertia = ensure_diag_inertia(
            np.median([item['inertia'] for item in candidates], axis=0)
        )
        fused_update_ratio = float(np.mean([item['update_ratio'] for item in candidates]))
        fused_stability = float(np.mean([item['stability_norm'] for item in candidates]))
        fused_tail_window = int(max(item['tail_window'] for item in candidates))
        candidates.append({
            'scheme': 'FUSED',
            'selection_origin': 'fused',
            'inertia': fused_inertia,
            'update_ratio': fused_update_ratio,
            'stability_norm': fused_stability,
            'tail_window': fused_tail_window,
            'excitation_balance': float(np.nanmean([item.get('excitation_balance', np.nan) for item in candidates])),
            'mean_axis_regressor_norms': np.nanmean(
                [item.get('mean_axis_regressor_norms', np.full(3, np.nan, dtype=float)) for item in candidates],
                axis=0,
            ),
            'source_schemes': [item['scheme'] for item in candidates if item.get('selection_origin') == 'identified'],
        })

    candidates.append({
        'scheme': 'NOMINAL',
        'selection_origin': 'fallback',
        'inertia': ensure_diag_inertia(fallback),
        'update_ratio': 0.0,
        'stability_norm': 0.0,
        'tail_window': 0,
        'excitation_balance': np.nan,
        'mean_axis_regressor_norms': np.full(3, np.nan, dtype=float),
        'source_schemes': [],
    })
    return candidates


def _evaluate_identified_inertia_candidate(
    candidate,
    dt,
    umax,
    true_inertia,
    validation_seeds,
    validation_T,
    use_star_tracker,
    excitation_profile,
):
    """
    对单个惯量候选做短时闭环鲁棒性验证。
    """
    candidate_inertia = ensure_diag_inertia(candidate['inertia'])
    max_omega_o = min(12.0, 0.49 / max(float(dt), 1e-6))
    validation_omega_c = min(4.5, 0.4 * max_omega_o)
    validation_omega_o = min(max_omega_o, max(2.2 * validation_omega_c, validation_omega_c + 2.0))
    adrc_params = _build_default_adrc_params(
        dt=dt,
        estimated_inertia=candidate_inertia,
        omega_c=validation_omega_c,
        omega_o=validation_omega_o,
    )

    validation_runs = []
    for seed in validation_seeds:
        validation_results = simulate_attitude_control(
            T=max(8.0, float(validation_T)),
            dt=dt,
            umax=umax,
            use_star_tracker=use_star_tracker,
            show_plots=False,
            controller_type='ADRC',
            adrc_params=adrc_params,
            seed=int(seed),
            excitation_profile=excitation_profile,
            true_inertia=true_inertia,
            verbose=False,
        )
        validation_runs.append({
            'seed': int(seed),
            'score': float(_score_adrc_results(validation_results)),
            'settle_time': float(validation_results['settle_time']),
            'final_error': float(validation_results['final_error']),
            'overshoot': float(validation_results['overshoot']),
            'effort': float(validation_results['effort']),
        })

    score_arr = np.array([item['score'] for item in validation_runs], dtype=float)
    settle_arr = np.array([item['settle_time'] for item in validation_runs], dtype=float)
    final_err_arr = np.array([item['final_error'] for item in validation_runs], dtype=float)
    overshoot_arr = np.array([item['overshoot'] for item in validation_runs], dtype=float)
    effort_arr = np.array([item['effort'] for item in validation_runs], dtype=float)

    update_penalty = 0.02 * max(0.0, 10.0 - float(candidate.get('update_ratio', 0.0)))
    stability_penalty = 25.0 * float(candidate.get('stability_norm', 0.0))
    robustness_penalty = 0.35 * float(np.std(score_arr))
    excitation_balance = candidate.get('excitation_balance', np.nan)
    excitation_penalty = (
        0.45 * max(0.0, 0.22 - float(excitation_balance))
        if np.isfinite(excitation_balance) else 0.0
    )
    fallback_penalty = 0.30 if str(candidate.get('selection_origin', 'identified')) == 'fallback' else 0.0
    model_score = float(
        np.mean(score_arr)
        + robustness_penalty
        + update_penalty
        + stability_penalty
        + excitation_penalty
        + fallback_penalty
    )

    evaluated = dict(candidate)
    evaluated.update({
        'validation_score': float(np.mean(score_arr)),
        'validation_score_std': float(np.std(score_arr)),
        'validation_score_best': float(np.min(score_arr)),
        'validation_settle_time': float(np.mean(settle_arr)),
        'validation_final_error': float(np.mean(final_err_arr)),
        'validation_overshoot': float(np.mean(overshoot_arr)),
        'validation_effort': float(np.mean(effort_arr)),
        'validation_runs': validation_runs,
        'validation_seed_count': int(len(validation_runs)),
        'excitation_penalty': float(excitation_penalty),
        'model_score': model_score,
    })
    return evaluated


def _select_identified_inertia(
    identification_results,
    dt,
    umax,
    fallback_inertia=None,
    true_inertia=None,
    validation_seed=0,
    validation_runs=3,
    validation_T=10.0,
    use_star_tracker=False,
    excitation_profile='aggressive',
):
    """
    从前置辨识结果中选择后续调参与控制所采用的名义惯量。

    改进后的策略:
    1. 对每种辨识结果做尾段平滑，减少“最后一个采样点偶然抖动”的影响
    2. 用统一 ADRC 基线参数对每个候选惯量做一次短时闭环验证
    3. 结合闭环验证分数、尾段稳定性与更新占比，自动选择更稳妥的模型
    """
    if fallback_inertia is None:
        fallback = np.array([0.50, 0.50, 0.50], dtype=float)
    else:
        fallback = ensure_diag_inertia(fallback_inertia)

    if not identification_results:
        return {
            'scheme': 'fallback',
            'inertia': fallback.copy(),
            'update_ratio': 0.0,
            'stability_norm': 0.0,
            'validation_score': np.inf,
            'selection_reason': 'no_identification_results',
        }

    validation_seeds = _resolve_validation_seeds(validation_seed, validation_runs)
    candidate_rows = []
    for candidate in _build_identification_selection_candidates(identification_results, fallback):
        candidate_rows.append(
            _evaluate_identified_inertia_candidate(
                candidate,
                dt=dt,
                umax=umax,
                true_inertia=true_inertia,
                validation_seeds=validation_seeds,
                validation_T=validation_T,
                use_star_tracker=use_star_tracker,
                excitation_profile=excitation_profile,
            )
        )

    if not candidate_rows:
        return {
            'scheme': 'fallback',
            'inertia': fallback.copy(),
            'update_ratio': 0.0,
            'stability_norm': 0.0,
            'validation_score': np.inf,
            'selection_reason': 'no_valid_candidates',
        }

    candidate_rows.sort(key=lambda item: item['model_score'])
    best = dict(candidate_rows[0])
    best['selection_reason'] = 'validated_candidate'
    best['candidates'] = candidate_rows
    return best


def compare_pd_gain_optimizers(
    bounds=((0.5, 0.05), (12.0, 4.0)),
    sim_cfg=None,
    weights=None,
    seed=0,
    show_plots=False,
    save_plots=True,
    output_dir=None,
    benchmark_runs=3,
    objective_eval_seeds=None,
    quick=False,  # 如果为True则使用更少迭代用于快速诊断
    parallel=False,  # 是否并行评估目标函数
    workers=4,      # 并行工作进程数
    verbose=True,
):
    """
    对 PD 控制器增益 (Kp, Kd) 进行自动调优，并横向对比多种优化算法。

    参数:
        bounds: ((Kp_min, Kd_min), (Kp_max, Kd_max))
        sim_cfg: 仿真参数字典（不含 Kp/Kd 与 controller_type）
        weights: 指标权重字典
        seed: 随机种子（保证各算法评估工况一致）
        show_plots: 是否弹窗显示图像
        save_plots: 是否保存图像到output_dir
        output_dir: 图像输出目录
        benchmark_runs: 每个优化器独立运行次数（用于稳健统计与小提琴分布）
        objective_eval_seeds: 目标函数评估所用随机种子列表；None时使用[seed]

    返回:
        comp: {
            'ranking': 排序后的结果列表,
            'best': 最优算法结果,
            'bounds': 搜索边界
        }
    """
    if sim_cfg is None:
        # 调参阶段使用较为完整的配置，保证性能指标有意义
        # 修复1: 延长仿真时间T到20.0s，使settle_time有区分度
        # 修复2: 统一用dt=0.03s保持采样精度
        sim_cfg = {
            'T': 20.0,           # 延长至20s以更好地观察稳定过程
            'dt': 0.03,          # 统一采样率（原来0.05太粗糙）
            'umax': 0.5,
            'use_star_tracker': False,
            'show_plots': False,
            'fov_deg': 20,
            'noise': 6e-4
        }
    else:
        sim_cfg = dict(sim_cfg)
        sim_cfg['show_plots'] = False
        # 修复: 统一整个优化过程的dt
        if 'dt' not in sim_cfg or sim_cfg['dt'] != 0.03:
            sim_cfg['dt'] = 0.03
    sim_cfg.setdefault('excitation_profile', 'aggressive')
    sim_cfg.setdefault('result_mode', 'metrics')

    if weights is None:
        # 修复5: 改进权重设置 - 使用非线性压缩后的合理权重
        # 注意: seats压缩后需要调整权重,使各指标有可比性
        weights = {
            'settle_time': 2.0,      # 加重settle_time,最关键的指标
            'overshoot': 1.0,        # 中等重要
            'final_error': 1.5,      # 较重要
            'effort': 0.5,           # 次要
            'sat_ratio': 1.0         # 中等重要(已×100)
        }

    lo = np.array(bounds[0], dtype=float)
    hi = np.array(bounds[1], dtype=float)
    cache = {}
    full_result_cache = {}
    oob_penalty_scale = 1e5  # 边界越界惩罚系数（与越界距离线性相关）
    out_dir = PROJECT_OUTPUT_DIR if output_dir is None else Path(output_dir)
    if objective_eval_seeds is None:
        eval_seeds = [int(seed), int(seed) + 1, int(seed) + 2]
    else:
        eval_seeds = [int(s) for s in objective_eval_seeds]
    if len(eval_seeds) == 0:
        eval_seeds = [int(seed)]
    risk_aversion = 0.35
    tail_weight = 0.15
    need_full_results = bool(show_plots or save_plots)
    if save_plots:
        out_dir.mkdir(parents=True, exist_ok=True)

    sim_cfg_full = dict(sim_cfg)
    sim_cfg_full['show_plots'] = False
    sim_cfg_full['result_mode'] = 'full'

    def fetch_full_results(Kp, Kd, sim_seed):
        """为少量入选参数补跑完整时序结果，用于论文风格图像输出。"""
        key = (round(float(Kp), 8), round(float(Kd), 8), int(sim_seed))
        if key not in full_result_cache:
            full_result_cache[key] = simulate_attitude_control(
                Kp=float(Kp),
                Kd=float(Kd),
                controller_type='PD',
                seed=int(sim_seed),
                verbose=False,
                **sim_cfg_full,
            )
        return full_result_cache[key]

    def objective(x):
        """目标函数（固定随机种子 + 越界惩罚）。"""
        x_raw = np.asarray(x, dtype=float)
        below = np.maximum(lo - x_raw, 0.0)
        above = np.maximum(x_raw - hi, 0.0)
        out_of_bounds = float(np.sum(below + above))
        oob_penalty = oob_penalty_scale * out_of_bounds

        x_clip = np.minimum(np.maximum(x_raw, lo), hi)
        Kp = float(x_clip[0])
        Kd = float(x_clip[1])
        key = (round(Kp, 8), round(Kd, 8))
        if key in cache:
            return float(cache[key]['score'] + oob_penalty)

        try:
            # 多随机种子平均，降低对单一样本序列的过拟合
            score_list = []
            metric_samples = []
            representative_results = None
            for eval_seed in eval_seeds:
                res = simulate_attitude_control(
                    Kp=Kp,
                    Kd=Kd,
                    controller_type='PD',
                    seed=eval_seed,
                    verbose=False,
                    **sim_cfg
                )
                s = _score_pd_results(res, weights)
                if not np.isfinite(s):
                    s = 1e9
                score_list.append(float(s))
                metric_samples.append(
                    (
                        float(res['settle_time']),
                        float(res['overshoot']),
                        float(res['final_error']),
                        float(res['effort']),
                        float(res['sat_ratio']),
                    )
                )
                if representative_results is None:
                    representative_results = res
            score_arr = np.asarray(score_list, dtype=float)
            metric_arr = np.asarray(metric_samples, dtype=float)
            score_mean = float(np.mean(score_arr))
            score_std = float(np.std(score_arr))
            score_tail = float(np.quantile(score_arr, 0.9))
            # 学术化处理: 风险敏感目标 = 均值 + 方差惩罚 + 尾部风险惩罚
            score = float(score_mean + risk_aversion * score_std + tail_weight * (score_tail - score_mean))
            metric_means = np.mean(metric_arr, axis=0)
            results = representative_results
        except Exception:
            results = None
            score_mean = 1e9
            score_std = 0.0
            score_tail = 1e9
            score = 1e9
            metric_means = np.array([np.inf, np.inf, np.inf, np.inf, np.inf], dtype=float)

        cache[key] = {
            'Kp': Kp,
            'Kd': Kd,
            'score': score,
            'score_mean': score_mean,
            'score_std': score_std,
            'score_tail': score_tail,
            'settle_time': float(metric_means[0]),
            'overshoot': float(metric_means[1]),
            'final_error': float(metric_means[2]),
            'effort': float(metric_means[3]),
            'sat_ratio': float(metric_means[4]),
            'results': results
        }
        return float(score + oob_penalty)

    def build_methods_runners(seed_base):
        x0_center = 0.5 * (lo + hi)
        if quick:
            return {
                'GridSearch': lambda obj: grid_search(obj, lo, hi, n_per_dim=4, parallel=parallel, workers=workers),
                'RandomSearch': lambda obj: random_search(obj, lo, hi, iters=24, seed=seed_base+1, parallel=parallel, workers=workers),
                'NelderMead': lambda obj: nelder_mead(obj, x0=x0_center.copy(), step=0.35*(hi-lo), iters=36, lo=lo, hi=hi),
                'SimAnneal': lambda obj: simulated_annealing(obj, lo, hi, iters=40, T0=10.0, decay=0.97, seed=seed_base+2),
                'PSO': lambda obj: pso(obj, lo, hi, iters=8, swarm=8, w=0.78, c1=1.5, c2=1.5, seed=seed_base+3, parallel=parallel, workers=workers)
            }
        return {
            # 预算尽量拉齐，避免个别优化器尚未展开搜索就提前结束。
            'GridSearch': lambda obj: grid_search(obj, lo, hi, n_per_dim=13, parallel=parallel, workers=workers),
            'RandomSearch': lambda obj: random_search(obj, lo, hi, iters=144, seed=seed_base+1, parallel=parallel, workers=workers),
            'NelderMead': lambda obj: nelder_mead(obj, x0=x0_center.copy(), step=0.25*(hi-lo), iters=120, lo=lo, hi=hi),
            'SimAnneal': lambda obj: simulated_annealing(obj, lo, hi, iters=144, T0=10.0, decay=0.97, seed=seed_base+2),
            'PSO': lambda obj: pso(obj, lo, hi, iters=12, swarm=12, w=0.72, c1=1.6, c2=1.6, seed=seed_base+3, parallel=parallel, workers=workers)
        }

    methods_runners = build_methods_runners(seed)
    method_names = list(methods_runners.keys())
    run_count = max(1, int(benchmark_runs))

    def _best_so_far(hist):
        arr = np.asarray(hist, dtype=float)
        if arr.size == 0:
            return np.array([np.inf], dtype=float)
        return np.minimum.accumulate(arr)

    method_run_data = {}
    comparison = []
    for m_idx, method_name in enumerate(method_names):
        run_records = []
        for run_idx in range(run_count):
            seed_run = seed + 1000 * (m_idx + 1) + 97 * run_idx
            runner = build_methods_runners(seed_run)[method_name]
            x_best, _f_best, hist = runner(objective)
            x_best = np.minimum(np.maximum(np.asarray(x_best, dtype=float), lo), hi)

            objective(x_best)
            rec = cache[(round(float(x_best[0]), 8), round(float(x_best[1]), 8))]
            if rec['results'] is None or not np.isfinite(rec['settle_time']):
                continue
            representative_results = rec['results']
            if need_full_results and (representative_results is None or 't' not in representative_results):
                representative_results = fetch_full_results(rec['Kp'], rec['Kd'], eval_seeds[0])
            run_records.append({
                'Kp': float(rec['Kp']),
                'Kd': float(rec['Kd']),
                'score': float(rec['score']),
                'score_mean': float(rec.get('score_mean', rec['score'])),
                'score_std_eval': float(rec.get('score_std', 0.0)),
                'score_tail': float(rec.get('score_tail', rec['score'])),
                'settle_time': float(rec['settle_time']),
                'overshoot': float(rec['overshoot']),
                'final_error': float(rec['final_error']),
                'effort': float(rec['effort']),
                'sat_ratio': float(rec['sat_ratio']),
                'history': _best_so_far(hist),
                'results': representative_results if need_full_results else None,
            })

        if not run_records:
            continue

        # 以均值-方差统计排序，同时保留该方法最佳一次参数作为代表
        score_arr = np.array([r['score'] for r in run_records], dtype=float)
        score_mean_arr = np.array([r['score_mean'] for r in run_records], dtype=float)
        score_std_eval_arr = np.array([r['score_std_eval'] for r in run_records], dtype=float)
        score_tail_arr = np.array([r['score_tail'] for r in run_records], dtype=float)
        settle_arr = np.array([r['settle_time'] for r in run_records], dtype=float)
        overshoot_arr = np.array([r['overshoot'] for r in run_records], dtype=float)
        final_err_arr = np.array([r['final_error'] for r in run_records], dtype=float)
        effort_arr = np.array([r['effort'] for r in run_records], dtype=float)
        sat_arr = np.array([r['sat_ratio'] for r in run_records], dtype=float)

        best_idx = int(np.argmin(score_arr))
        best_run = run_records[best_idx]

        hist_list = [r['history'] for r in run_records]
        max_len = max(h.size for h in hist_list)
        hist_mat = np.vstack([
            np.pad(h, (0, max_len - h.size), mode='edge') if h.size < max_len else h
            for h in hist_list
        ])
        mean_hist = np.mean(hist_mat, axis=0)
        std_hist = np.std(hist_mat, axis=0)

        method_run_data[method_name] = {
            'runs': run_records,
            'score_dist': score_arr,
            'mean_hist': mean_hist,
            'std_hist': std_hist,
            'best_run': best_run
        }

        comparison.append({
            'method': method_name,
            'Kp': best_run['Kp'],
            'Kd': best_run['Kd'],
            'score': float(np.mean(score_arr)),
            'score_std': float(np.std(score_arr)),
            'score_best': float(np.min(score_arr)),
            'score_mean_eval': float(np.mean(score_mean_arr)),
            'score_std_eval': float(np.mean(score_std_eval_arr)),
            'score_tail_eval': float(np.mean(score_tail_arr)),
            'settle_time': float(np.mean(settle_arr)),
            'overshoot': float(np.mean(overshoot_arr)),
            'final_error': float(np.mean(final_err_arr)),
            'effort': float(np.mean(effort_arr)),
            'sat_ratio': float(np.mean(sat_arr)),
            'history': mean_hist
        })

    comparison = sorted(comparison, key=lambda d: d['score'])
    best = comparison[0] if len(comparison) > 0 else None

    if verbose:
        print(f"\n{'='*78}")
        print("PD增益自动调优：优化器横向对比（Kp, Kd）")
        print(f"{'='*78}")
        print(f"{'算法':<14} {'Kp':>8} {'Kd':>8} {'Score(mean)':>12} {'Score(std)':>11} {'EvalStd':>10} {'Tail90':>10} {'Ts(s)':>9}")
        print("-" * 112)
        for row in comparison:
            print(
                f"{row['method']:<14} "
                f"{row['Kp']:>8.3f} {row['Kd']:>8.3f} {row['score']:>12.3f} "
                f"{row['score_std']:>11.3f} {row['score_std_eval']:>10.3f} "
                f"{row['score_tail_eval']:>10.3f} {row['settle_time']:>9.3f}"
            )

        if best is not None:
            print("-" * 95)
            print(
                f"最优算法: {best['method']} | Kp={best['Kp']:.4f}, Kd={best['Kd']:.4f}, "
                f"Score(mean)={best['score']:.4f}, Score(best)={best['score_best']:.4f}"
            )

    if save_plots and len(comparison) > 0:
        ranking_rows = []
        for rank_idx, row in enumerate(comparison, start=1):
            ranking_rows.append({
                'rank': rank_idx,
                'method': row['method'],
                'Kp': f"{row['Kp']:.6f}",
                'Kd': f"{row['Kd']:.6f}",
                'score_mean': f"{row['score']:.6f}",
                'score_std_between_runs': f"{row['score_std']:.6f}",
                'score_best': f"{row['score_best']:.6f}",
                'score_eval_mean': f"{row['score_mean_eval']:.6f}",
                'score_eval_std': f"{row['score_std_eval']:.6f}",
                'score_eval_tail90': f"{row['score_tail_eval']:.6f}",
                'settle_time_s': f"{row['settle_time']:.6f}",
                'overshoot_deg': f"{row['overshoot']:.6f}",
                'final_error_deg': f"{row['final_error']:.6f}",
                'effort_Nms': f"{row['effort']:.6f}",
                'sat_ratio': f"{row['sat_ratio']:.6f}",
            })
        _write_csv_rows(
            out_dir / 'optimizer_ranking.csv',
            list(ranking_rows[0].keys()),
            ranking_rows,
        )

        cache_rows = []
        for (kp_val, kd_val), rec in sorted(cache.items()):
            cache_rows.append({
                'Kp': f"{kp_val:.8f}",
                'Kd': f"{kd_val:.8f}",
                'score': f"{float(rec['score']):.8f}",
                'score_mean': f"{float(rec.get('score_mean', rec['score'])):.8f}",
                'score_std': f"{float(rec.get('score_std', 0.0)):.8f}",
                'score_tail90': f"{float(rec.get('score_tail', rec['score'])):.8f}",
            })
        _write_csv_rows(
            out_dir / 'optimizer_evaluated_points.csv',
            list(cache_rows[0].keys()) if cache_rows else ['Kp', 'Kd', 'score'],
            cache_rows,
        )

    # 优化器图像输出（核心结果版）
    if (show_plots or save_plots) and len(comparison) > 0:
        if save_plots:
            out_dir.mkdir(parents=True, exist_ok=True)

        # 1) 已评估点构成的参数景观图
        fig_landscape = plot_gain_landscape_from_cache(
            cache,
            (float(lo[0]), float(hi[0])),
            (float(lo[1]), float(hi[1])),
            title='PD增益优化景观图（基于已评估点的风险敏感目标）',
        )
        if fig_landscape is not None and save_plots:
            _save_figure(fig_landscape, out_dir / "pd_optimizer_landscape.png")
        elif fig_landscape is not None:
            plt.close(fig_landscape)

        # 2) 优化综合仪表板
        fig_report = plot_optimizer_report_dashboard(
            cache,
            method_run_data,
            comparison,
            bounds=(lo, hi),
            title='PD增益优化综合仪表板',
        )
        if save_plots:
            _save_figure(fig_report, out_dir / "pd_optimizer_report_dashboard.png")
        elif fig_report is not None:
            plt.close(fig_report)

        fig_convergence = plot_optimizer_convergence_statistics(
            method_run_data,
            comparison,
            title=f'优化器收敛统计曲线（均值±标准差，n={run_count}）',
        )
        if fig_convergence is not None and save_plots:
            _save_figure(fig_convergence, out_dir / "pd_optimizer_convergence_statistics.png")
        elif fig_convergence is not None:
            plt.close(fig_convergence)

        fig_response = plot_optimizer_best_response_dashboard(
            method_run_data,
            comparison,
            title='不同优化器最优增益下的闭环响应对比',
        )
        if fig_response is not None and save_plots:
            _save_figure(fig_response, out_dir / "pd_optimizer_best_response_dashboard.png")
        elif fig_response is not None:
            plt.close(fig_response)

        fig_tradeoff = plot_optimizer_tradeoff_scatter(
            comparison,
            title='优化器精度-能耗权衡图',
        )
        if fig_tradeoff is not None and save_plots:
            _save_figure(fig_tradeoff, out_dir / "pd_optimizer_tradeoff_scatter.png")
        elif fig_tradeoff is not None:
            plt.close(fig_tradeoff)

        fig_heatmap = plot_optimizer_metric_heatmap(
            comparison,
            title='优化器指标热力图（列内归一化）',
        )
        if fig_heatmap is not None and save_plots:
            _save_figure(fig_heatmap, out_dir / "pd_optimizer_metric_heatmap.png")
        elif fig_heatmap is not None:
            plt.close(fig_heatmap)


    return {
        'ranking': comparison,
        'best': best,
        'bounds': (lo.copy(), hi.copy()),
        'output_dir': str(out_dir.resolve()) if save_plots else None,
        'cache_size': len(cache),
    }


def tune_adrc_bandwidths(
    estimated_inertia,
    sim_cfg=None,
    output_dir=None,
    objective_eval_seeds=None,
    verbose=True,
):
    """
    基于已辨识惯量，对 ADRC 的 omega_c / omega_o 做轻量自动整定。
    """
    estimated_inertia = ensure_diag_inertia(estimated_inertia)
    if sim_cfg is None:
        sim_cfg = {
            'T': 12.0,
            'dt': 0.03,
            'umax': 0.5,
            'use_star_tracker': False,
            'show_plots': False,
            'excitation_profile': 'aggressive',
        }
    else:
        sim_cfg = dict(sim_cfg)
        sim_cfg['show_plots'] = False
        sim_cfg.setdefault('excitation_profile', 'aggressive')

    dt = float(sim_cfg.get('dt', 0.03))
    out_dir = None if output_dir is None else Path(output_dir)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    if objective_eval_seeds is None:
        eval_seeds = [0, 1]
    else:
        eval_seeds = [int(s) for s in objective_eval_seeds] or [0]

    max_omega_o = min(12.0, 0.49 / max(dt, 1e-6))
    coarse_wc = np.array([3.2, 4.2, 5.2, 6.2, 7.2, 8.0, 8.8], dtype=float)
    coarse_ratio = np.array([1.05, 1.15, 1.30, 1.55, 1.90], dtype=float)
    cache = {}

    def evaluate_candidate(omega_c, ratio):
        omega_c = float(omega_c)
        ratio = float(ratio)
        omega_o = float(min(max_omega_o, max(omega_c + 0.2, omega_c * ratio)))
        key = (round(omega_c, 6), round(omega_o, 6))
        if key in cache:
            return cache[key]

        per_seed = []
        metric_samples = []
        for eval_seed in eval_seeds:
            adrc_params = _build_default_adrc_params(
                dt=dt,
                estimated_inertia=estimated_inertia,
                omega_c=omega_c,
                omega_o=omega_o,
            )
            results = simulate_attitude_control(
                controller_type='ADRC',
                adrc_params=adrc_params,
                seed=eval_seed,
                verbose=False,
                **sim_cfg,
            )
            per_seed.append(_score_adrc_results(results))
            metric_samples.append(
                (
                    float(results['settle_time']),
                    float(results['overshoot']),
                    float(results['final_error']),
                    float(results['IAE']),
                    float(results['ITAE']),
                    float(results['ISU']),
                    float(results['effort']),
                    float(results['sat_ratio']),
                )
            )

        score_arr = np.asarray(per_seed, dtype=float)
        metric_arr = np.asarray(metric_samples, dtype=float)
        metric_means = np.mean(metric_arr, axis=0)
        score_mean = float(np.mean(score_arr))
        score_std = float(np.std(score_arr))
        score_tail = float(np.quantile(score_arr, 0.9))
        score = float(score_mean + 0.35 * score_std + 0.15 * (score_tail - score_mean))

        rec = {
            'omega_c': omega_c,
            'omega_o': omega_o,
            'omega_ratio': omega_o / max(omega_c, 1e-9),
            'score': score,
            'score_mean': score_mean,
            'score_std': score_std,
            'score_tail': score_tail,
            'settle_time': float(metric_means[0]),
            'overshoot': float(metric_means[1]),
            'final_error': float(metric_means[2]),
            'IAE': float(metric_means[3]),
            'ITAE': float(metric_means[4]),
            'ISU': float(metric_means[5]),
            'effort': float(metric_means[6]),
            'sat_ratio': float(metric_means[7]),
        }
        cache[key] = rec
        return rec

    candidates = []
    for omega_c in coarse_wc:
        for ratio in coarse_ratio:
            candidates.append(evaluate_candidate(omega_c, ratio))
    candidates.sort(key=lambda item: item['score'])

    best_coarse = candidates[0]
    fine_wc = np.clip(
        np.array(
            [
                best_coarse['omega_c'] - 0.8,
                best_coarse['omega_c'] - 0.4,
                best_coarse['omega_c'],
                best_coarse['omega_c'] + 0.4,
                best_coarse['omega_c'] + 0.8,
            ],
            dtype=float,
        ),
        2.8,
        min(9.2, max_omega_o - 0.2),
    )
    fine_ratio = np.clip(
        np.array(
            [
                best_coarse['omega_ratio'] - 0.15,
                best_coarse['omega_ratio'] - 0.08,
                best_coarse['omega_ratio'],
                best_coarse['omega_ratio'] + 0.08,
                best_coarse['omega_ratio'] + 0.15,
            ],
            dtype=float,
        ),
        1.05,
        2.20,
    )
    for omega_c in np.unique(np.round(fine_wc, 4)):
        for ratio in np.unique(np.round(fine_ratio, 4)):
            candidates.append(evaluate_candidate(float(omega_c), float(ratio)))

    ranking = sorted(
        cache.values(),
        key=lambda item: item['score']
    )
    best = dict(ranking[0]) if ranking else None

    if out_dir is not None and ranking:
        rows = []
        for idx, row in enumerate(ranking, start=1):
            rows.append({
                'rank': idx,
                'omega_c': f"{row['omega_c']:.6f}",
                'omega_o': f"{row['omega_o']:.6f}",
                'omega_ratio': f"{row['omega_ratio']:.6f}",
                'score': f"{row['score']:.6f}",
                'score_mean': f"{row['score_mean']:.6f}",
                'score_std': f"{row['score_std']:.6f}",
                'score_tail90': f"{row['score_tail']:.6f}",
                'settle_time_s': f"{row['settle_time']:.6f}",
                'overshoot_deg': f"{row['overshoot']:.6f}",
                'final_error_deg': f"{row['final_error']:.6f}",
                'IAE_deg_s': f"{row['IAE']:.6f}",
                'ITAE_deg_s2': f"{row['ITAE']:.6f}",
                'ISU_N2m2s': f"{row['ISU']:.6f}",
                'effort_Nms': f"{row['effort']:.6f}",
                'sat_ratio': f"{row['sat_ratio']:.6f}",
            })
        _write_csv_rows(
            out_dir / 'adrc_tuning_ranking.csv',
            list(rows[0].keys()),
            rows,
        )

    if verbose and best is not None:
        print(f"\n{'='*78}")
        print("ADRC 带宽自动整定结果（omega_c, omega_o）")
        print(f"{'='*78}")
        print(
            f"最优参数: omega_c={best['omega_c']:.4f}, omega_o={best['omega_o']:.4f}, "
            f"ratio={best['omega_ratio']:.3f}, score={best['score']:.4f}, "
            f"Ts={best['settle_time']:.3f}s, Final={best['final_error']:.4f}deg"
        )

    return {
        'ranking': ranking,
        'best': best,
        'output_dir': str(out_dir.resolve()) if out_dir is not None else None,
        'evaluated_points': len(cache),
    }


def main(profile='full'):
    """主函数 - 同时对比PD和ADRC控制器"""
    print("=" * 70)
    print("卫星姿态控制仿真系统 - PD vs ADRC 性能对比")
    print("=" * 70)

    run_profile = _resolve_run_profile(profile)

    # 仿真参数
    T = run_profile['T']
    dt = run_profile['dt']
    umax = 0.5  # 最大控制力矩 (N·m)
    true_inertia_diag = np.array([1.60, 1.40, 1.00], dtype=float)

    print("\n共同仿真参数:")
    print(f"  运行档位 profile = {run_profile['profile']}")
    print(f"  仿真时间 T = {T} s")
    print(f"  时间步长 dt = {dt} s")
    print(f"  最大力矩 umax = {umax} N·m")
    print(f"  真实惯量 J = {true_inertia_diag} kg*m^2")
    output_layout = _make_output_layout(PROJECT_OUTPUT_DIR)
    output_dir = output_layout['root']

    # ====== 前置动力学参数辨识 ======
    identification_results = _run_inertia_identification_cases(run_profile, umax, output_layout, true_inertia_diag)
    selected_inertia_summary = _select_identified_inertia(
        identification_results,
        dt=dt,
        umax=umax,
        fallback_inertia=np.diag(np.asarray(CONFIG['spacecraft']['inertia'], dtype=float)),
        true_inertia=true_inertia_diag,
        validation_seed=run_profile['objective_eval_seeds'][0] if run_profile.get('objective_eval_seeds') else 0,
        validation_T=max(8.0, float(run_profile['T'])),
        use_star_tracker=run_profile['use_star_tracker'],
        excitation_profile='auto',
    )
    estimated_inertia = selected_inertia_summary['inertia']
    print("\n前置辨识完成，后续流程将使用该名义惯量:")
    print(
        f"  采用方案 = {selected_inertia_summary['scheme']}, "
        f"I = {estimated_inertia}, "
        f"update_ratio = {selected_inertia_summary['update_ratio']:.1f}%, "
        f"validation_score = {selected_inertia_summary['validation_score']:.4f}"
    )

    # ====== 基于辨识模型的控制器调参 ======
    print(f"\n{'='*70}")
    print("基于前置辨识结果进行控制器调参与配置...")
    print(f"{'='*70}")
    tuning = compare_pd_gain_optimizers(
        bounds=((1.0, 0.1), (8.0, 2.0)),
        sim_cfg={
            'T': 12.0,
            'dt': 0.03,
            'umax': umax,
            'use_star_tracker': False,
            'fov_deg': 20,
            'noise': 6e-4,
            'true_inertia': true_inertia_diag,
        },
        seed=0,
        show_plots=False,
        save_plots=True,
        output_dir=str(output_layout['optimization']),
        benchmark_runs=run_profile['optimizer_runs'],
        objective_eval_seeds=run_profile['objective_eval_seeds'],
        quick=run_profile['optimizer_quick']
    )

    adrc_tuning = tune_adrc_bandwidths(
        estimated_inertia=estimated_inertia,
        sim_cfg={
            'T': float(run_profile['T']),
            'dt': float(dt),
            'umax': umax,
            'use_star_tracker': run_profile['use_star_tracker'],
            'show_plots': False,
            'excitation_profile': 'auto',
            'true_inertia': true_inertia_diag,
        },
        output_dir=str(output_layout['optimization_adrc']),
        objective_eval_seeds=run_profile['objective_eval_seeds'],
        verbose=True,
    )

    # ====== 运行 PD 控制器仿真 ======
    print(f"\n{'='*70}")
    print("运行 PD 控制器仿真...")
    print(f"{'='*70}")
    if tuning['best'] is not None:
        Kp = float(tuning['best']['Kp'])
        Kd = float(tuning['best']['Kd'])
        print(f"自动调参结果: Kp = {Kp:.4f}, Kd = {Kd:.4f}")
    else:
        Kp = 6.0
        Kd = 1.5
        print("自动调参失败，回退到默认参数。")
    print(f"PD参数: Kp = {Kp}, Kd = {Kd}")
    
    results_pd = simulate_attitude_control(
        Kp=Kp,
        Kd=Kd,
        T=T,
        dt=dt,
        umax=umax,
        use_star_tracker=run_profile['use_star_tracker'],
        show_plots=False,
        controller_type='PD',
        true_inertia=true_inertia_diag,
    )
    _save_simulation_plot_suite(results_pd, output_layout['simulation_pd'], 'PD', detail_level=OUTPUT_DETAIL_LEVEL)
    print("PD仿真完成！")
    
    # ====== 运行 ADRC 控制器仿真 ======
    print(f"\n{'='*70}")
    print("运行 ADRC 控制器仿真...")
    print(f"{'='*70}")
    best_adrc = adrc_tuning.get('best') if adrc_tuning is not None else None
    tuned_omega_c = float(best_adrc['omega_c']) if best_adrc is not None else 4.0
    tuned_omega_o = float(best_adrc['omega_o']) if best_adrc is not None else 8.0
    adrc_params = _build_default_adrc_params(
        dt=dt,
        estimated_inertia=estimated_inertia,
        omega_c=tuned_omega_c,
        omega_o=tuned_omega_o,
    )
    print(f"ADRC参数: omega_c = {adrc_params['omega_c']} rad/s, omega_o = {adrc_params['omega_o']} rad/s")
    print(f"         b0 = {adrc_params['b0']}")
    
    results_adrc = simulate_attitude_control(
        T=T,
        dt=dt,
        umax=umax,
        use_star_tracker=run_profile['use_star_tracker'],
        show_plots=False,
        controller_type='ADRC',
        adrc_params=adrc_params,
        true_inertia=true_inertia_diag,
    )
    _save_simulation_plot_suite(results_adrc, output_layout['simulation_adrc'], 'ADRC', detail_level=OUTPUT_DETAIL_LEVEL)
    print("ADRC仿真完成！")
    
    # ====== 性能对比 ======
    _print_controller_comparison(results_pd, results_adrc)
    
    # ====== 绘制对比图 ======
    _save_controller_comparison_summary(results_pd, results_adrc, output_layout['comparison'])
    if run_profile['save_comparison_plots']:
        _save_controller_comparison_plots(results_pd, results_adrc, umax, output_layout['comparison'])
    else:
        _remove_comparison_plot_files(output_layout['comparison'])
        print("\n快速模式已启用：跳过对比图生成。")

    report_path = _generate_markdown_report(
        tuning=tuning,
        results_pd=results_pd,
        results_adrc=results_adrc,
        output_layout=output_layout,
        run_profile=run_profile,
        identification_results=identification_results,
        selected_inertia_summary=selected_inertia_summary,
        adrc_tuning=adrc_tuning,
    )

    print("\n输出目录结构:")
    print(f"  根目录: {output_dir.resolve()}")
    print(f"  优化图: {output_layout['optimization'].resolve()}")
    print(f"  ADRC整定: {output_layout['optimization_adrc'].resolve()}")
    print(f"  PD仿真图: {output_layout['simulation_pd'].resolve()}")
    print(f"  ADRC仿真图: {output_layout['simulation_adrc'].resolve()}")
    print(f"  对比图: {output_layout['comparison'].resolve()}")
    print(f"  RLS辨识图: {output_layout['identification_rls'].resolve()}")
    print(f"  MEKF辨识图: {output_layout['identification_mekf'].resolve()}")
    print(f"  Markdown报告: {report_path.resolve()}")

    print(f"\n{'='*70}")
    print("仿真对比完成！")
    print(f"{'='*70}")
    
    return results_pd, results_adrc


if __name__ == "__main__":
    results_pd, results_adrc = main()
