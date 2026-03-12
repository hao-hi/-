"""
卫星姿态控制仿真主程序
整合动力学、控制器、MEKF、星敏感器等模块
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
import io
import contextlib
from pathlib import Path
from datetime import datetime
from config import CONFIG
from core_utils import quat_from_axis_angle, quat_to_R, quat_mul, quat_angle_error_rad
from dynamics import Spacecraft
from controller import pd_torque
from adrc_controller import ADRCController, adrc_torque
from startracker import StarTracker
from mekf import MEKFBiasOnly, MEKF_Augmented
from estimators import InertiaRLS, ensure_diag_inertia
from 优化器.optimizers import (
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
    plot_pareto_front,
    plot_gain_metrics_heatmap,
    plot_optimizer_report_dashboard,
    plot_optimizer_score_distribution,
    plot_optimizer_parameter_scatter,
    plot_multiple_responses,
    plot_simulation_report_dashboard,
    plot_phase_portrait
)

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DETAIL_LEVEL = "core"


def _make_output_layout(root_dir):
    """
    创建统一的输出目录结构。
    """
    root = Path(root_dir)
    layout = {
        'root': root,
        'optimization': root / 'optimization',
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


def _build_default_adrc_params(dt):
    """
    构造默认 ADRC 参数，并按估计惯量匹配 b0。
    """
    estimated_inertia = np.array([0.05, 0.05, 0.05], dtype=float)
    return {
        'b0': 1.0 / estimated_inertia,
        'omega_c': 2.5,
        'omega_o': 8.0,
        'dt': dt,
    }


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


def _compute_settle_time(err_hist, t_hist, strict_threshold=2.0, trans_threshold=3.0):
    """
    采用两阶段规则计算调节时间。
    """
    settle_time = t_hist[-1] + 10.0
    stable_window = max(20, int(len(err_hist) * 0.05))

    for i in range(len(err_hist) - stable_window):
        if np.all(err_hist[i:i + stable_window] <= strict_threshold):
            return float(t_hist[i])

    for i in range(len(err_hist) - 1, -1, -1):
        if err_hist[i] > trans_threshold:
            return float(t_hist[i] + 1.0)

    return float(settle_time)


def _assemble_simulation_results(history, dist_const, settle_time, sat_count, umax, controller_kind, controller_type, Kp, Kd, inertia_scheme, inertia_reference=None):
    """
    汇总仿真历史与性能指标。
    """
    err_hist = history['err']
    t_hist = history['t']
    u_hist = history['u']

    overshoot = float(np.max(err_hist) - err_hist[-1])
    final_error = float(err_hist[-1])
    effort = float(np.trapz(np.linalg.norm(u_hist, axis=1), t_hist))
    sat_ratio = sat_count / len(t_hist)

    results = dict(history)
    results.update({
        'dist_const': dist_const.copy(),
        'settle_time': settle_time,
        'overshoot': overshoot,
        'final_error': final_error,
        'effort': effort,
        'sat_ratio': sat_ratio,
        'inertia_estimator_scheme': inertia_scheme,
        'Kp': Kp if controller_kind == 'PD' else None,
        'Kd': Kd if controller_kind == 'PD' else None,
        'controller_type': controller_type,
        'u_limit': float(umax),
        'inertia_reference': None if inertia_reference is None else ensure_diag_inertia(inertia_reference),
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


def _save_simulation_data_tables(results, output_dir, controller_kind):
    """
    保存单次仿真的摘要指标与时序数据，便于后续论文制表或二次分析。
    """
    controller_tag = str(controller_kind).upper()
    metrics_rows = [
        {'metric': 'controller_type', 'value': results.get('controller_type', controller_tag), 'unit': '-'},
        {'metric': 'settle_time', 'value': f"{results['settle_time']:.6f}", 'unit': 's'},
        {'metric': 'overshoot', 'value': f"{results['overshoot']:.6f}", 'unit': 'deg'},
        {'metric': 'final_error', 'value': f"{results['final_error']:.6f}", 'unit': 'deg'},
        {'metric': 'effort', 'value': f"{results['effort']:.6f}", 'unit': 'N·m·s'},
        {'metric': 'sat_ratio', 'value': f"{results['sat_ratio']:.6f}", 'unit': '-'},
        {'metric': 'u_limit', 'value': f"{results['u_limit']:.6f}", 'unit': 'N·m'},
    ]
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


def _run_inertia_identification_cases(run_profile, umax, output_layout):
    """
    运行两组动力学参数辨识案例，并保存专用图像与数据。
    """
    print(f"\n{'='*70}")
    print("运行动力学参数辨识案例...")
    print(f"{'='*70}")

    base_cfg = {
        'T': max(12.0, float(run_profile['T'])),
        'dt': float(run_profile['dt']),
        'umax': float(umax),
        'use_star_tracker': False,
        'show_plots': False,
        'seed': 7,
    }

    rls_results = simulate_attitude_control(
        controller_type='ADRC',
        adrc_params={
            'b0': 1.0 / np.array([0.05, 0.05, 0.05], dtype=float),
            'omega_c': 2.5,
            'omega_o': 8.0,
            'dt': base_cfg['dt'],
        },
        inertia_estimator_cfg={
            'scheme': 'RLS',
            'J0': np.array([0.060, 0.060, 0.045], dtype=float),
            'lambda_factor': 0.995,
            'min_inertia': 0.02,
            'max_inertia': 0.12,
            'min_accel_excitation': 8e-4,
            'min_torque_excitation': 8e-4,
            'max_update_step': 8e-4,
        },
        **base_cfg,
    )
    _save_simulation_plot_suite(rls_results, output_layout['identification_rls'], 'ADRC', detail_level=OUTPUT_DETAIL_LEVEL)

    mekf_results = simulate_attitude_control(
        controller_type='ADRC',
        adrc_params={
            'b0': 1.0 / np.array([0.05, 0.05, 0.05], dtype=float),
            'omega_c': 2.5,
            'omega_o': 8.0,
            'dt': base_cfg['dt'],
        },
        inertia_estimator_cfg={
            'scheme': 'MEKF',
            'J0': np.array([0.060, 0.060, 0.045], dtype=float),
            'min_inertia': 0.02,
            'max_inertia': 0.12,
        },
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


def _print_controller_comparison(results_pd, results_adrc):
    """
    打印 PD 与 ADRC 的核心性能对比。
    """
    print(f"\n{'='*70}")
    print("性能对比结果")
    print(f"{'='*70}")
    print(f"\n{'指标':<20} {'PD控制器':<20} {'ADRC控制器':<20} {'改进百分比':<15}")
    print("-" * 75)

    settle_impr = (results_pd['settle_time'] - results_adrc['settle_time']) / results_pd['settle_time'] * 100
    print(f"{'调节时间 (s)':<20} {results_pd['settle_time']:<20.2f} {results_adrc['settle_time']:<20.2f} {settle_impr:>13.1f}%")

    overshoot_impr = (results_pd['overshoot'] - results_adrc['overshoot']) / (results_pd['overshoot'] + 1e-6) * 100
    print(f"{'超调量 (度)':<20} {results_pd['overshoot']:<20.2f} {results_adrc['overshoot']:<20.2f} {overshoot_impr:>13.1f}%")

    final_error_impr = (results_pd['final_error'] - results_adrc['final_error']) / (results_pd['final_error'] + 1e-6) * 100
    print(f"{'稳态误差 (度)':<20} {results_pd['final_error']:<20.3f} {results_adrc['final_error']:<20.3f} {final_error_impr:>13.1f}%")
    print("\n[关键指标] Steady-state Error（稳态误差）")
    print(f"  PD   : {results_pd['final_error']:.4f} deg")
    print(f"  ADRC : {results_adrc['final_error']:.4f} deg")
    if results_adrc['final_error'] < results_pd['final_error']:
        print("  结论 : ADRC 在常值干扰下静差更小（符合抗扰预期）。")
    else:
        print("  结论 : 当前参数下 ADRC 静差未优于 PD，建议继续提升 omega_o 或延长仿真时间。")

    effort_impr = (results_pd['effort'] - results_adrc['effort']) / results_pd['effort'] * 100
    print(f"{'控制能耗':<20} {results_pd['effort']:<20.3f} {results_adrc['effort']:<20.3f} {effort_impr:>13.1f}%")

    sat_impr = (results_pd['sat_ratio'] - results_adrc['sat_ratio']) / (results_pd['sat_ratio'] + 1e-6) * 100
    print(f"{'饱和比例 (%)':<20} {results_pd['sat_ratio']*100:<20.1f} {results_adrc['sat_ratio']*100:<20.1f} {sat_impr:>13.1f}%")


def _save_controller_comparison_summary(results_pd, results_adrc, output_dir):
    """
    保存 PD 与 ADRC 的对比指标表，便于直接插入报告。
    """
    rows = []
    metrics = [
        ('settle_time', '调节时间', 's'),
        ('overshoot', '超调量', 'deg'),
        ('final_error', '稳态误差', 'deg'),
        ('effort', '控制能耗', 'N·m·s'),
        ('sat_ratio', '饱和比例', '-'),
    ]
    for key, name, unit in metrics:
        pd_val = float(results_pd[key])
        adrc_val = float(results_adrc[key])
        improvement = (pd_val - adrc_val) / (pd_val + 1e-12) * 100.0
        rows.append({
            'metric_key': key,
            'metric_name': name,
            'unit': unit,
            'pd_value': f"{pd_val:.6f}",
            'adrc_value': f"{adrc_val:.6f}",
            'improvement_pct': f"{improvement:.3f}",
        })
    _write_csv_rows(
        Path(output_dir) / 'controller_comparison_summary.csv',
        ['metric_key', 'metric_name', 'unit', 'pd_value', 'adrc_value', 'improvement_pct'],
        rows,
    )


def _generate_markdown_report(tuning, results_pd, results_adrc, output_layout, run_profile, identification_results=None):
    """
    自动生成 Markdown 风格结果报告，串联关键图表和指标。
    """
    root_dir = Path(output_layout['root'])
    report_path = root_dir / 'simulation_summary_report.md'
    generated_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    comparison_rows = []
    metrics = [
        ('调节时间', 'settle_time', 's'),
        ('超调量', 'overshoot', 'deg'),
        ('稳态误差', 'final_error', 'deg'),
        ('控制能耗', 'effort', 'N·m·s'),
        ('饱和比例', 'sat_ratio', '-'),
    ]
    for display_name, key, unit in metrics:
        pd_val = float(results_pd[key])
        adrc_val = float(results_adrc[key])
        improvement = (pd_val - adrc_val) / (pd_val + 1e-12) * 100.0
        comparison_rows.append([
            display_name,
            f"{pd_val:.4f}",
            f"{adrc_val:.4f}",
            unit,
            f"{improvement:+.2f}%",
        ])

    ranking_rows = []
    for idx, row in enumerate(tuning.get('ranking', []), start=1):
        ranking_rows.append([
            idx,
            row['method'],
            f"{row['Kp']:.4f}",
            f"{row['Kd']:.4f}",
            f"{row['score']:.4f}",
            f"{row['settle_time']:.4f}",
            f"{row['final_error']:.4f}",
            f"{row['effort']:.4f}",
        ])

    best_method = tuning.get('best')
    best_summary = "无"
    if best_method is not None:
        best_summary = (
            f"{best_method['method']} (Kp={best_method['Kp']:.4f}, "
            f"Kd={best_method['Kd']:.4f}, score={best_method['score']:.4f})"
        )

    image_blocks = [
        ("优化综合仪表板", output_layout['optimization'] / 'pd_optimizer_report_dashboard.png'),
        ("优化景观图", output_layout['optimization'] / 'pd_optimizer_landscape.png'),
        ("PD 单次仿真综合仪表板", output_layout['simulation_pd'] / 'simulation_report_dashboard.png'),
        ("ADRC 单次仿真综合仪表板", output_layout['simulation_adrc'] / 'simulation_report_dashboard.png'),
        ("控制器性能对比图", output_layout['comparison'] / 'pd_vs_adrc_comparison.png'),
        ("控制器时域叠加图", output_layout['comparison'] / 'pd_vs_adrc_time_response_overlay.png'),
    ]
    if identification_results:
        image_blocks.extend([
            ("RLS 参数辨识综合图", output_layout['identification_rls'] / 'inertia_identification_dashboard.png'),
            ("MEKF 参数辨识综合图", output_layout['identification_mekf'] / 'inertia_identification_dashboard.png'),
        ])

    lines = [
        "# 卫星姿态控制仿真结果汇总报告",
        "",
        f"- 生成时间: `{generated_time}`",
        f"- 运行档位: `{run_profile['profile']}`",
        f"- 仿真时长: `{run_profile['T']}` s",
        f"- 时间步长: `{run_profile['dt']}` s",
        f"- 优化最优方法: `{best_summary}`",
        "",
        "## 1. 结论摘要",
        "",
        f"- PD 控制器末端姿态误差: `{results_pd['final_error']:.4f} deg`",
        f"- ADRC 控制器末端姿态误差: `{results_adrc['final_error']:.4f} deg`",
        f"- PD 控制器调节时间: `{results_pd['settle_time']:.4f} s`",
        f"- ADRC 控制器调节时间: `{results_adrc['settle_time']:.4f} s`",
        f"- PD 控制器控制能耗: `{results_pd['effort']:.4f} N·m·s`",
        f"- ADRC 控制器控制能耗: `{results_adrc['effort']:.4f} N·m·s`",
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
        id_rows = []
        for scheme_name, res in identification_results.items():
            inertia_ref = np.asarray(res.get('inertia_reference', res['inertia_est'][0]), dtype=float)
            inertia_final = np.asarray(res['inertia_est'][-1], dtype=float)
            err_norm = float(np.linalg.norm(inertia_final - inertia_ref))
            update_ratio = float(np.mean(np.asarray(res.get('inertia_update_mask', []), dtype=float) > 0.5) * 100.0)
            id_rows.append([
                scheme_name,
                f"[{inertia_ref[0]:.4f}, {inertia_ref[1]:.4f}, {inertia_ref[2]:.4f}]",
                f"[{inertia_final[0]:.4f}, {inertia_final[1]:.4f}, {inertia_final[2]:.4f}]",
                f"{err_norm:.4e}",
                f"{update_ratio:.1f}%",
            ])
        lines.extend([
            _markdown_table(
                ['方案', '参考惯量', '最终估计', '最终误差范数', '更新占比'],
                id_rows,
            ),
            "",
        ])
    else:
        lines.extend([
            "本次运行未生成参数辨识案例。",
            "",
        ])

    lines.extend([
        "## 6. 结果文件索引",
        "",
    ])

    file_index_rows = [
        ['优化结果', _relative_markdown_path(root_dir, output_layout['optimization']), '优化图像、已评估点与排序表'],
        ['PD 仿真', _relative_markdown_path(root_dir, output_layout['simulation_pd']), 'PD 单次仿真图像与时序数据'],
        ['ADRC 仿真', _relative_markdown_path(root_dir, output_layout['simulation_adrc']), 'ADRC 单次仿真图像与时序数据'],
        ['对比结果', _relative_markdown_path(root_dir, output_layout['comparison']), 'PD vs ADRC 对比图和摘要表'],
    ]
    if identification_results:
        file_index_rows.extend([
            ['RLS 辨识', _relative_markdown_path(root_dir, output_layout['identification_rls']), 'RLS 参数辨识图像与时序数据'],
            ['MEKF 辨识', _relative_markdown_path(root_dir, output_layout['identification_mekf']), '增广 MEKF 参数辨识图像与时序数据'],
        ])

    lines.extend([
        _markdown_table(
            ['类别', '目录', '说明'],
            file_index_rows,
        ),
        "",
        "## 7. 附注",
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
    print(f"\n生成并保存对比图表...")
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
                               inertia_estimator_cfg=None):
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
    
    返回:
        results: 包含仿真结果的字典
    """
    # 固定随机种子，保证 PD 与 ADRC 在各自仿真中经历完全一致的随机序列
    np.random.seed(seed)
    controller_kind = controller_type.upper()
    
    # 初始化系统
    sc = Spacecraft(umax=umax)
    st = StarTracker(n_stars_catalog=1600, fov_deg=fov_deg, dir_noise_std=noise) if use_star_tracker else None
    
    # 初始化控制器
    if controller_kind == 'ADRC':
        if adrc_params is None:
            adrc_params = _build_default_adrc_params(dt)
        adrc_controller = ADRCController(**adrc_params)
        print(f"使用ADRC控制器: omega_c={adrc_params.get('omega_c', 2.5)}, omega_o={adrc_params.get('omega_o', 8.0)}")
    else:
        adrc_controller = None
        print(f"使用PD控制器: Kp={Kp}, Kd={Kd}")
    
    # 初始条件
    # 修复: 激进地增加初始条件扰度，使不同增益能有最显著的差异
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    # 更激进的初始姿态偏角：15度→45度（调参时），强力激发控制器性能差异
    # T > 15说明是调参模式（快速调参用T=20），T <= 15是验证模式
    initial_angle = np.deg2rad(45.0 if T > 15 else 15.0)  # 调参模式用45度！
    q = quat_from_axis_angle(axis, initial_angle)
    # 更激进的初始角速度，使系统处于高度动态状态
    w = np.deg2rad(np.array([1.0, -1.0, 0.5]))  # 从[0.5,-0.5,0.3]→[1.0,-1.0,0.5]
    qd = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)  # 期望姿态（单位四元数）
    
    # 初始化 MEKF / 增广 MEKF 与在线惯量辨识器
    initial_J_diag = np.diag(sc.J).copy()
    inertia_reference = initial_J_diag.copy()
    inertia_scheme, inertia_identifier = _build_inertia_identifier(
        inertia_estimator_cfg,
        initial_J_diag=initial_J_diag,
        initial_q=q,
    )
    if inertia_scheme == 'MEKF':
        mekf = inertia_identifier
    else:
        mekf = MEKFBiasOnly()
        mekf.q = q.copy()
    use_augmented_mekf = isinstance(mekf, MEKF_Augmented)
    
    steps = int(T / dt)
    history = _allocate_simulation_history(steps)
    sat_count = 0
    u_prev = np.zeros(3)  # 用于ADRC的上一时刻控制量
    w_filt_prev = None
    current_J_diag = initial_J_diag.copy()

    # 外部扰动：常值项 + 小幅随机噪声
    # 常值项用于体现 ADRC 对偏置扰动（如重力梯度/气动不平衡）的抑制能力
    dist_const = np.array([0.02, -0.015, 0.01], dtype=float)
    dist_noise_std = 0.003
    # 陀螺仪真实偏置模型：常值初始偏置 + 随机游走
    true_bias = np.array([0.01, -0.015, 0.005], dtype=float)
    gyro_bias_rw_std = 1e-5
    gyro_noise_std = 0.002

    # 批量预生成循环内会反复用到的随机量，减少 Python 层频繁调用开销。
    fallback_noise_axes = np.random.randn(steps, 3)
    fallback_startracker_angles = np.abs(np.random.normal(0, noise, size=steps))
    fallback_ideal_angles = np.abs(np.random.normal(0, 0.001, size=steps))
    gyro_bias_rw_noise = np.random.randn(steps, 3)
    gyro_measurement_noise = np.random.randn(steps, 3)
    dist_noise_samples = np.random.randn(steps, 3)
    
        # 主仿真循环
    for k in range(steps):
        t = k * dt
        history['t'][k] = t
        
        # 获取真实姿态
        Rtrue = quat_to_R(q) if (use_star_tracker and st is not None) else None
        
        # 星敏感器观测
        if use_star_tracker and st is not None:
            q_meas = st.observe(Rtrue)
            if q_meas is None:
                # 如果星敏感器失效，使用真实姿态（加噪声）
                q_meas = _apply_quaternion_noise(q, fallback_noise_axes[k], fallback_startracker_angles[k])
        else:
            # 理想测量（加小噪声）
            q_meas = _apply_quaternion_noise(q, fallback_noise_axes[k], fallback_ideal_angles[k])
        
        # 真实陀螺偏置随机游走，并生成含偏置测量值
        true_bias += gyro_bias_rw_std * gyro_bias_rw_noise[k]
        w_gyro = w + true_bias + gyro_noise_std * gyro_measurement_noise[k]
        history['gyro_bias_true'][k] = true_bias
        if use_augmented_mekf:
            mekf.predict(w_gyro, dt, u_applied=u_prev)
        else:
            mekf.predict(w_gyro, dt)
        
        # MEKF更新
        mekf.update(q_meas)

        # 近似角加速度：使用当前与上一拍滤波角速度做离散差分
        w_filt = w_gyro - mekf.b
        if w_filt_prev is None:
            wdot_est = np.zeros(3, dtype=float)
        else:
            wdot_est = (w_filt - w_filt_prev) / dt
        w_filt_prev = w_filt.copy()

        # 在线惯量辨识接口：先更新 J，再同步给动力学模型与 ADRC 的 b0
        if inertia_scheme == 'RLS' and inertia_identifier is not None:
            dist_torque_est = (
                adrc_controller.get_disturbance_estimate_torque()
                if adrc_controller is not None else np.zeros(3, dtype=float)
            )
            J_diag_est, inertia_updated = inertia_identifier.update(
                omega=w_filt,
                wdot=wdot_est,
                u=u_prev,
                disturbance_torque=dist_torque_est,
            )
            current_J_diag = _sync_inertia_estimate(J_diag_est, adrc_controller)
        elif inertia_scheme == 'MEKF' and use_augmented_mekf:
            current_J_diag = _sync_inertia_estimate(mekf.get_inertia_diag(), adrc_controller)
            inertia_updated = True
        else:
            current_J_diag = np.diag(sc.J).copy()
            inertia_updated = False
        
        # 控制器计算（先使用真实姿态 q 做控制，验证控制器极限性能）
        if controller_kind == 'ADRC':
            u, q_e = adrc_torque(qd, q, w, adrc_controller, dt, u_prev)
            # 为和动力学中的 dist（力矩）同量纲，使用等效力矩形式的扰动估计
            dist_est = adrc_controller.get_disturbance_estimate_torque()
        else:
            u, q_e = pd_torque(qd, q, w, Kp, Kd)
            dist_est = np.zeros(3, dtype=float)
        
        # 外部扰动：常值 + 随机白噪声
        dist = dist_const + dist_noise_std * dist_noise_samples[k]
        
        # 动力学积分
        q, w, u_sat = sc.step(q, w, u, dt, dist=dist)
        
        # 更新上一时刻控制量（用于ADRC）
        u_prev = u_sat.copy()
        
        # 记录数据
        history['q_true'][k] = q
        history['q_est'][k] = mekf.q
        history['w'][k] = w
        history['u'][k] = u_sat
        history['dist_true'][k] = dist
        history['dist_est'][k] = dist_est
        history['gyro_bias_est'][k] = mekf.b
        history['inertia_est'][k] = current_J_diag
        history['inertia_update_mask'][k] = 1.0 if inertia_updated else 0.0
        history['wdot_est'][k] = wdot_est
        
        # 计算姿态误差（使用真实姿态与期望姿态之间的误差，作为控制性能指标）
        history['err'][k] = np.rad2deg(quat_angle_error_rad(qd, q))
        
        # 统计饱和次数
        if np.any(np.abs(u_sat) >= (umax - 1e-6)):
            sat_count += 1
    
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
        inertia_reference=inertia_reference,
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
    max_err = float(np.max(np.asarray(results.get('err', [final_error]), dtype=float)))
    
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
    workers=4       # 并行工作进程数
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
    if save_plots:
        out_dir.mkdir(parents=True, exist_ok=True)

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
            results_last = None
            for eval_seed in eval_seeds:
                with contextlib.redirect_stdout(io.StringIO()):
                    res = simulate_attitude_control(
                        Kp=Kp,
                        Kd=Kd,
                        controller_type='PD',
                        seed=eval_seed,
                        **sim_cfg
                    )
                s = _score_pd_results(res, weights)
                if not np.isfinite(s):
                    s = 1e9
                score_list.append(float(s))
                results_last = res
            score_arr = np.asarray(score_list, dtype=float)
            score_mean = float(np.mean(score_arr))
            score_std = float(np.std(score_arr))
            score_tail = float(np.quantile(score_arr, 0.9))
            # 学术化处理: 风险敏感目标 = 均值 + 方差惩罚 + 尾部风险惩罚
            score = float(score_mean + risk_aversion * score_std + tail_weight * (score_tail - score_mean))
            results = results_last
        except Exception as e:
            results = None
            score_mean = 1e9
            score_std = 0.0
            score_tail = 1e9
            score = 1e9

        cache[key] = {
            'Kp': Kp,
            'Kd': Kd,
            'score': score,
            'score_mean': score_mean,
            'score_std': score_std,
            'score_tail': score_tail,
            'results': results
        }
        return float(score + oob_penalty)

    def build_methods_runners(seed_base):
        x0_center = 0.5 * (lo + hi)
        if quick:
            return {
                'GridSearch': lambda obj: grid_search(obj, lo, hi, n_per_dim=3, parallel=parallel, workers=workers),
                'RandomSearch': lambda obj: random_search(obj, lo, hi, iters=12, seed=seed_base+1, parallel=parallel, workers=workers),
                'NelderMead': lambda obj: nelder_mead(obj, x0=x0_center.copy(), step=0.4*(hi-lo), iters=20, lo=lo, hi=hi),
                'SimAnneal': lambda obj: simulated_annealing(obj, lo, hi, iters=20, T0=10.0, decay=0.97, seed=seed_base+2),
                'PSO': lambda obj: pso(obj, lo, hi, iters=10, swarm=6, w=0.8, c1=1.5, c2=1.5, seed=seed_base+3, parallel=parallel, workers=workers)
            }
        return {
            # 近似统一预算: Grid(11x11=121), Random(120), SA(120), PSO(12x10=120)
            'GridSearch': lambda obj: grid_search(obj, lo, hi, n_per_dim=11, parallel=parallel, workers=workers),
            'RandomSearch': lambda obj: random_search(obj, lo, hi, iters=120, seed=seed_base+1, parallel=parallel, workers=workers),
            'NelderMead': lambda obj: nelder_mead(obj, x0=x0_center.copy(), step=0.25*(hi-lo), iters=80, lo=lo, hi=hi),
            'SimAnneal': lambda obj: simulated_annealing(obj, lo, hi, iters=120, T0=10.0, decay=0.97, seed=seed_base+2),
            'PSO': lambda obj: pso(obj, lo, hi, iters=10, swarm=12, w=0.72, c1=1.6, c2=1.6, seed=seed_base+3, parallel=parallel, workers=workers)
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
            if rec['results'] is None:
                continue
            results = rec['results']
            run_records.append({
                'Kp': float(rec['Kp']),
                'Kd': float(rec['Kd']),
                'score': float(rec['score']),
                'score_mean': float(rec.get('score_mean', rec['score'])),
                'score_std_eval': float(rec.get('score_std', 0.0)),
                'score_tail': float(rec.get('score_tail', rec['score'])),
                'settle_time': float(results['settle_time']),
                'overshoot': float(results['overshoot']),
                'final_error': float(results['final_error']),
                'effort': float(results['effort']),
                'sat_ratio': float(results['sat_ratio']),
                'history': _best_so_far(hist),
                'results': results
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


    return {
        'ranking': comparison,
        'best': best,
        'bounds': (lo.copy(), hi.copy()),
        'output_dir': str(out_dir.resolve()) if save_plots else None,
        'cache_size': len(cache),
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
    
    # 物理模型匹配：估计卫星惯性张量（kg·m²）
    ESTIMATED_INERTIA = np.array([0.05, 0.05, 0.05])
    b0_estimated = 1.0 / ESTIMATED_INERTIA
    
    print(f"\n共同仿真参数:")
    print(f"  运行档位 profile = {run_profile['profile']}")
    print(f"  仿真时间 T = {T} s")
    print(f"  时间步长 dt = {dt} s")
    print(f"  最大力矩 umax = {umax} N·m")
    print(f"  卫星惯性 I = {ESTIMATED_INERTIA}")
    output_layout = _make_output_layout(PROJECT_OUTPUT_DIR)
    output_dir = output_layout['root']

    # ====== 五优化器 PD 增益调参与对比（精简核心图） ======
    tuning = compare_pd_gain_optimizers(
        bounds=((1.0, 0.1), (8.0, 2.0)),
        sim_cfg={
            'T': 12.0,
            'dt': 0.03,
            'umax': umax,
            'use_star_tracker': False,
            'fov_deg': 20,
            'noise': 6e-4
        },
        seed=0,
        show_plots=False,
        save_plots=True,
        output_dir=str(output_layout['optimization']),
        benchmark_runs=run_profile['optimizer_runs'],
        objective_eval_seeds=run_profile['objective_eval_seeds'],
        quick=run_profile['optimizer_quick']
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
        controller_type='PD'
    )
    _save_simulation_plot_suite(results_pd, output_layout['simulation_pd'], 'PD', detail_level=OUTPUT_DETAIL_LEVEL)
    print("PD仿真完成！")
    
    # ====== 运行 ADRC 控制器仿真 ======
    print(f"\n{'='*70}")
    print("运行 ADRC 控制器仿真...")
    print(f"{'='*70}")
    adrc_params = {
        'b0': b0_estimated,         # 物理匹配的控制增益
        'omega_c': 2.5,             # 控制器带宽（rad/s）
        'omega_o': 8.0,             # 观测器带宽（rad/s）
        'dt': dt                    # 采样时间（用于稳定性检查）
    }
    print(f"ADRC参数: omega_c = {adrc_params['omega_c']} rad/s, omega_o = {adrc_params['omega_o']} rad/s")
    print(f"         b0 = {adrc_params['b0']}")
    
    results_adrc = simulate_attitude_control(
        T=T,
        dt=dt,
        umax=umax,
        use_star_tracker=run_profile['use_star_tracker'],
        show_plots=False,
        controller_type='ADRC',
        adrc_params=adrc_params
    )
    _save_simulation_plot_suite(results_adrc, output_layout['simulation_adrc'], 'ADRC', detail_level=OUTPUT_DETAIL_LEVEL)
    print("ADRC仿真完成！")

    identification_results = _run_inertia_identification_cases(run_profile, umax, output_layout)
    
    # ====== 性能对比 ======
    _print_controller_comparison(results_pd, results_adrc)
    
    # ====== 绘制对比图 ======
    if run_profile['save_comparison_plots']:
        _save_controller_comparison_plots(results_pd, results_adrc, umax, output_layout['comparison'])
    else:
        print("\n快速模式已启用：跳过对比图生成。")

    report_path = _generate_markdown_report(
        tuning=tuning,
        results_pd=results_pd,
        results_adrc=results_adrc,
        output_layout=output_layout,
        run_profile=run_profile,
        identification_results=identification_results,
    )

    print("\n输出目录结构:")
    print(f"  根目录: {output_dir.resolve()}")
    print(f"  优化图: {output_layout['optimization'].resolve()}")
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
