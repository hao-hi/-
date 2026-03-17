"""
可视化模块
提供各种绘图功能
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
from core_utils import quat_angle_errors_deg

# 解决中文和负号显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'sans-serif']
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'STSong']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 120
matplotlib.rcParams['savefig.dpi'] = 260
matplotlib.rcParams['savefig.facecolor'] = 'white'
matplotlib.rcParams['axes.titleweight'] = 'semibold'
matplotlib.rcParams['axes.labelsize'] = 11.5
matplotlib.rcParams['axes.linewidth'] = 0.9
matplotlib.rcParams['axes.axisbelow'] = True
matplotlib.rcParams['xtick.labelsize'] = 9.8
matplotlib.rcParams['ytick.labelsize'] = 9.8
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.major.size'] = 4.2
matplotlib.rcParams['ytick.major.size'] = 4.2
matplotlib.rcParams['xtick.minor.size'] = 2.4
matplotlib.rcParams['ytick.minor.size'] = 2.4
matplotlib.rcParams['legend.frameon'] = True
matplotlib.rcParams['legend.edgecolor'] = '#c7d0da'
matplotlib.rcParams['legend.facecolor'] = 'white'
matplotlib.rcParams['legend.framealpha'] = 0.94
matplotlib.rcParams['legend.fancybox'] = False
matplotlib.rcParams['grid.alpha'] = 0.85


PLOT_COLORS = {
    'pd': '#1f4e79',
    'adrc': '#b64a4a',
    'truth': '#1f252d',
    'estimate': '#2e7d6f',
    'x': '#c35a49',
    'y': '#2f8b7c',
    'z': '#4f6d8a',
    'grid': '#d9dfe7',
    'limit': '#7a8087',
    'accent': '#c98a38',
    'success': '#5f8d57',
    'warning': '#cc6f2c',
    'soft_fill': '#eef3f8',
}
_TRAPEZOID = getattr(np, "trapezoid", np.trapz)


def _curve_integral(y, x):
    """统一的梯形积分兼容层。"""
    return float(_TRAPEZOID(y, x))


def _style_axes(ax, xlabel=None, ylabel=None, title=None):
    """统一坐标轴样式。"""
    ax.set_facecolor('#fcfdff')
    ax.grid(True, color=PLOT_COLORS['grid'], linewidth=0.82, alpha=0.82)
    ax.minorticks_on()
    ax.grid(which='minor', color=PLOT_COLORS['grid'], linewidth=0.45, alpha=0.38)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#8e98a5')
    ax.spines['bottom'].set_color('#8e98a5')
    ax.spines['left'].set_linewidth(0.95)
    ax.spines['bottom'].set_linewidth(0.95)
    ax.tick_params(colors='#33404d', which='major', width=0.85)
    ax.tick_params(colors='#66717d', which='minor', width=0.55)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title, fontsize=13.2, fontweight='semibold', pad=11, color='#1f2a36')


def _finish_figure(fig, title=None):
    """统一图像收尾。"""
    fig.patch.set_facecolor('white')
    if title:
        fig.suptitle(title, fontsize=15.8, fontweight='semibold', y=0.992, color='#17212b')
        fig.tight_layout(rect=(0, 0, 1, 0.975))
    else:
        fig.tight_layout()
    fig.align_ylabels()
    return fig


def _panel_label(ax, label):
    """在子图左上角添加论文风格面板编号。"""
    ax.text(
        0.0,
        1.02,
        label,
        transform=ax.transAxes,
        ha='left',
        va='bottom',
        fontsize=10.8,
        fontweight='semibold',
        color='#223243',
        bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='#c7d0da', alpha=0.97),
    )


def _metric_box(ax, lines, loc='upper right'):
    """在图中添加简洁指标框。"""
    if isinstance(lines, str):
        lines = [lines]
    text = '\n'.join(lines)
    anchor_map = {
        'upper right': (0.98, 0.98, 'right', 'top'),
        'upper left': (0.02, 0.98, 'left', 'top'),
        'lower right': (0.98, 0.04, 'right', 'bottom'),
        'lower left': (0.02, 0.04, 'left', 'bottom'),
    }
    x, y, ha, va = anchor_map.get(loc, anchor_map['upper right'])
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=8.9,
        color='#243447',
        linespacing=1.35,
        bbox=dict(boxstyle='round,pad=0.34', facecolor='white', edgecolor='#c7d0da', alpha=0.95),
    )


def _apply_time_axis(ax, t_hist):
    """统一时间轴范围与刻度密度。"""
    t_arr = np.asarray(t_hist, dtype=float)
    if t_arr.size == 0:
        return
    ax.set_xlim(float(t_arr[0]), float(t_arr[-1]))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(6))


def _controller_color(controller_type):
    """根据控制器类型返回主色。"""
    if str(controller_type).upper() == 'ADRC':
        return PLOT_COLORS['adrc']
    return PLOT_COLORS['pd']


def _annotate_series_end(ax, x, y, text, color):
    """在曲线终点添加简洁标注。"""
    ax.scatter([x], [y], s=28, color=color, zorder=6)
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(6, 6),
        textcoords='offset points',
        fontsize=8,
        color=color,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, alpha=0.85),
    )


def _highlight_active_spans(ax, t_hist, mask, color, alpha=0.08, min_width=0.0):
    """
    将更新/激活区间以纵向浅色带标出，增强时域图的叙事性。
    """
    t_arr = np.asarray(t_hist, dtype=float)
    mask_arr = np.asarray(mask, dtype=float).reshape(-1) > 0.5
    if t_arr.size == 0 or mask_arr.size != t_arr.size or not np.any(mask_arr):
        return

    starts = []
    ends = []
    in_span = False
    start_idx = 0
    for idx, active in enumerate(mask_arr):
        if active and not in_span:
            in_span = True
            start_idx = idx
        elif not active and in_span:
            in_span = False
            starts.append(start_idx)
            ends.append(idx - 1)
    if in_span:
        starts.append(start_idx)
        ends.append(t_arr.size - 1)

    for s_idx, e_idx in zip(starts, ends):
        x0 = float(t_arr[s_idx])
        x1 = float(t_arr[e_idx])
        if x1 - x0 < min_width and e_idx + 1 < t_arr.size:
            x1 = float(t_arr[e_idx + 1])
        ax.axvspan(x0, x1, color=color, alpha=alpha, linewidth=0.0, zorder=0)


def plot_attitude_estimation(t_hist, q_true_seq, q_est_seq, title="姿态估计结果"):
    """
    绘制姿态估计结果（四元数分量）
    
    参数:
        t_hist: 时间序列
        q_true_seq: 真实姿态四元数序列
        q_est_seq: 估计姿态四元数序列
        title: 图标题
    """
    fig, axs = plt.subplots(4, 1, figsize=(10.5, 10.5), sharex=True)
    labels = ['q0', 'q1', 'q2', 'q3']
    rmse = np.sqrt(np.mean((q_true_seq - q_est_seq) ** 2, axis=0))
    
    for k in range(4):
        axs[k].plot(t_hist, q_true_seq[:, k], linestyle='--', color=PLOT_COLORS['truth'], label=f"真实 {labels[k]}", linewidth=2.0)
        axs[k].plot(t_hist, q_est_seq[:, k], color=PLOT_COLORS['estimate'], label=f"估计 {labels[k]}", linewidth=1.8)
        _style_axes(axs[k], ylabel=labels[k])
        _apply_time_axis(axs[k], t_hist)
        _panel_label(axs[k], f'({chr(ord("a") + k)})')
        axs[k].legend(loc='upper right')
        axs[k].fill_between(t_hist, q_true_seq[:, k], q_est_seq[:, k], color=PLOT_COLORS['estimate'], alpha=0.08)
        _metric_box(axs[k], [f'RMSE = {rmse[k]:.4e}'], loc='lower right')
    
    _style_axes(axs[-1], xlabel='时间 (s)')
    return _finish_figure(fig, title)


def plot_attitude_error(t_hist, q_true_seq, q_est_seq, title="姿态误差"):
    """
    绘制姿态误差（角度）
    
    参数:
        t_hist: 时间序列
        q_true_seq: 真实姿态四元数序列
        q_est_seq: 估计姿态四元数序列
        title: 图标题
    """
    err_hist = quat_angle_errors_deg(q_true_seq, q_est_seq)
    
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 5.2))
    ax.plot(t_hist, err_hist, color=PLOT_COLORS['pd'], linewidth=2.2, label='姿态误差')
    ax.fill_between(t_hist, err_hist, color=PLOT_COLORS['pd'], alpha=0.12)
    ax.axhline(1.0, color=PLOT_COLORS['accent'], linestyle='--', linewidth=1.2, label='1度参考线')
    _style_axes(ax, xlabel='时间 (s)', ylabel='姿态误差 (度)', title=title)
    _apply_time_axis(ax, t_hist)
    _annotate_series_end(ax, t_hist[-1], err_hist[-1], f"{err_hist[-1]:.2f} deg", PLOT_COLORS['pd'])
    _metric_box(
        ax,
        [
            f'峰值误差 = {np.max(err_hist):.2f} deg',
            f'均值误差 = {np.mean(err_hist):.2f} deg',
            f'末端误差 = {err_hist[-1]:.2f} deg',
        ],
        loc='upper right',
    )
    ax.legend()
    return _finish_figure(fig)


def plot_angular_velocity(t_hist, w_hist, title="角速度"):
    """
    绘制角速度
    
    参数:
        t_hist: 时间序列
        w_hist: 角速度序列 (N×3)
        title: 图标题
    """
    fig, axs = plt.subplots(3, 1, figsize=(10.5, 8.5), sharex=True)
    labels = [('ωx', PLOT_COLORS['x']), ('ωy', PLOT_COLORS['y']), ('ωz', PLOT_COLORS['z'])]
    
    for k, (label, color) in enumerate(labels):
        series = np.rad2deg(w_hist[:, k])
        axs[k].plot(t_hist, series, color=color, linewidth=1.8, label=label)
        axs[k].axhline(0.0, color='#9aa5b1', linewidth=0.9, linestyle=':')
        _style_axes(axs[k], ylabel=f'{label} (deg/s)')
        _apply_time_axis(axs[k], t_hist)
        _panel_label(axs[k], f'({chr(ord("a") + k)})')
        _annotate_series_end(axs[k], t_hist[-1], series[-1], f"{series[-1]:.2f}", color)
        _metric_box(axs[k], [f'RMS = {np.sqrt(np.mean(series ** 2)):.2f}'], loc='lower right')
        axs[k].legend(loc='upper right')
    
    _style_axes(axs[-1], xlabel='时间 (s)')
    return _finish_figure(fig, title)


def plot_control_torque(t_hist, u_hist, umax=None, title="控制力矩"):
    """
    绘制控制力矩
    
    参数:
        t_hist: 时间序列
        u_hist: 控制力矩序列 (N×3)
        umax: 最大力矩限制（可选，用于绘制饱和线）
        title: 图标题
    """
    fig, axs = plt.subplots(4, 1, figsize=(10.5, 10.5), sharex=True)
    labels = [('ux', PLOT_COLORS['x']), ('uy', PLOT_COLORS['y']), ('uz', PLOT_COLORS['z'])]
    
    for k, (label, color) in enumerate(labels):
        axs[k].plot(t_hist, u_hist[:, k], color=color, linewidth=1.8, label=label)
        if umax is not None:
            axs[k].axhline(y=umax, color=PLOT_COLORS['limit'], linestyle='--', alpha=0.8, label=f'限制 ±{umax}')
            axs[k].axhline(y=-umax, color=PLOT_COLORS['limit'], linestyle='--', alpha=0.8)
            sat_mask = np.abs(u_hist[:, k]) >= 0.98 * umax
            if np.any(sat_mask):
                axs[k].fill_between(t_hist, u_hist[:, k], where=sat_mask, color=PLOT_COLORS['warning'], alpha=0.18)
        _style_axes(axs[k], ylabel=f'{label} (N·m)')
        _apply_time_axis(axs[k], t_hist)
        _panel_label(axs[k], f'({chr(ord("a") + k)})')
        axs[k].legend(loc='upper right')
    
    # 绘制力矩幅值
    u_norm = np.linalg.norm(u_hist, axis=1)
    axs[3].plot(t_hist, u_norm, color=PLOT_COLORS['truth'], linewidth=2.2, label='||u||')
    if umax is not None:
        axs[3].axhline(y=umax, color=PLOT_COLORS['limit'], linestyle='--', alpha=0.8, label=f'限制 {umax}')
    _style_axes(axs[3], xlabel='时间 (s)', ylabel='||u|| (N·m)')
    _apply_time_axis(axs[3], t_hist)
    _panel_label(axs[3], '(d)')
    _annotate_series_end(axs[3], t_hist[-1], u_norm[-1], f"{u_norm[-1]:.3f}", PLOT_COLORS['truth'])
    _metric_box(axs[3], [f'积分能耗 = {_curve_integral(u_norm, t_hist):.3f} N·m·s'], loc='upper right')
    axs[3].legend(loc='upper right')
    return _finish_figure(fig, title)


def plot_control_response(t_hist, err_hist, w_hist, u_hist, umax=None, title="控制系统响应"):
    """
    绘制完整的控制系统响应
    
    参数:
        t_hist: 时间序列
        err_hist: 姿态误差序列（角度，度）
        w_hist: 角速度序列 (N×3)
        u_hist: 控制力矩序列 (N×3)
        umax: 最大力矩限制
        title: 图标题
    """
    fig = plt.figure(figsize=(12, 10.2))
    
    # 姿态误差
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t_hist, err_hist, color=PLOT_COLORS['pd'], linewidth=2.2)
    ax1.fill_between(t_hist, err_hist, color=PLOT_COLORS['pd'], alpha=0.10)
    ax1.axhline(1.0, color=PLOT_COLORS['accent'], linestyle='--', linewidth=1.1)
    _style_axes(ax1, ylabel='姿态误差 (度)', title=title)
    _apply_time_axis(ax1, t_hist)
    _panel_label(ax1, '(a)')
    _metric_box(ax1, [f'峰值 = {np.max(err_hist):.2f} deg', f'末端 = {err_hist[-1]:.2f} deg'], loc='upper right')
    
    # 角速度
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(t_hist, np.rad2deg(w_hist[:, 0]), color=PLOT_COLORS['x'], label='ωx', linewidth=1.8)
    ax2.plot(t_hist, np.rad2deg(w_hist[:, 1]), color=PLOT_COLORS['y'], label='ωy', linewidth=1.8)
    ax2.plot(t_hist, np.rad2deg(w_hist[:, 2]), color=PLOT_COLORS['z'], label='ωz', linewidth=1.8)
    _style_axes(ax2, ylabel='角速度 (deg/s)')
    _apply_time_axis(ax2, t_hist)
    _panel_label(ax2, '(b)')
    ax2.legend()
    
    # 控制力矩
    ax3 = plt.subplot(3, 1, 3)
    u_norm = np.linalg.norm(u_hist, axis=1)
    ax3.plot(t_hist, u_norm, color=PLOT_COLORS['truth'], linewidth=2.2, label='||u||')
    if umax is not None:
        ax3.axhline(y=umax, color=PLOT_COLORS['limit'], linestyle='--', alpha=0.8, label=f'限制 {umax}')
    _style_axes(ax3, xlabel='时间 (s)', ylabel='||u|| (N·m)')
    _apply_time_axis(ax3, t_hist)
    _panel_label(ax3, '(c)')
    _metric_box(ax3, [f'能耗 = {_curve_integral(u_norm, t_hist):.3f} N·m·s'], loc='upper right')
    ax3.legend()
    return _finish_figure(fig)


def plot_observer_tracking(t_hist, truth_hist, estimate_hist, title, ylabel, truth_label='真实值', estimate_label='估计值'):
    """
    绘制三轴真值/估计值对比图，适合扰动或偏置估计。
    """
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.2), sharex=True)
    axis_names = ['x', 'y', 'z']
    axis_colors = [PLOT_COLORS['x'], PLOT_COLORS['y'], PLOT_COLORS['z']]

    for i, ax in enumerate(axes):
        ax.plot(t_hist, truth_hist[:, i], color=PLOT_COLORS['truth'], linewidth=1.9, label=truth_label)
        ax.plot(t_hist, estimate_hist[:, i], color=axis_colors[i], linestyle='--', linewidth=1.8, label=estimate_label)
        _style_axes(ax, ylabel=f'{axis_names[i]}轴{ylabel}')
        _apply_time_axis(ax, t_hist)
        _panel_label(ax, f'({chr(ord("a") + i)})')
        axis_rmse = np.sqrt(np.mean((truth_hist[:, i] - estimate_hist[:, i]) ** 2))
        _metric_box(ax, [f'RMSE = {axis_rmse:.4e}'], loc='lower right')
        if i == 0:
            ax.legend(loc='upper right')

    _style_axes(axes[-1], xlabel='时间 (s)')
    return _finish_figure(fig, title)


def plot_controller_comparison_dashboard(results_pd, results_adrc, umax, title="PD控制器 vs ADRC控制器性能对比"):
    """
    绘制 PD 与 ADRC 的四联对比图。
    """
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5))
    t = np.asarray(results_pd['t'], dtype=float)
    pd_settle = float(results_pd.get('settle_time', t[-1]))
    adrc_settle = float(results_adrc.get('settle_time', t[-1]))

    ax = axes[0, 0]
    err_pd = np.asarray(results_pd['err'], dtype=float)
    err_adrc = np.asarray(results_adrc['err'], dtype=float)
    ax.fill_between(t, err_pd, color=PLOT_COLORS['pd'], alpha=0.08)
    ax.fill_between(t, err_adrc, color=PLOT_COLORS['adrc'], alpha=0.08)
    ax.plot(t, err_pd, color=PLOT_COLORS['pd'], linewidth=2.35, label='PD')
    ax.plot(t, err_adrc, color=PLOT_COLORS['adrc'], linewidth=2.35, linestyle='--', label='ADRC')
    ax.axhline(y=1.0, color=PLOT_COLORS['accent'], linestyle=':', alpha=0.9, label='1度误差界')
    ax.axvline(pd_settle, color=PLOT_COLORS['pd'], linestyle=':', linewidth=1.0, alpha=0.7)
    ax.axvline(adrc_settle, color=PLOT_COLORS['adrc'], linestyle=':', linewidth=1.0, alpha=0.7)
    _style_axes(ax, xlabel='时间 (s)', ylabel='姿态误差 (度)', title='姿态误差对比')
    _apply_time_axis(ax, t)
    _panel_label(ax, '(a)')
    _annotate_series_end(ax, t[-1], err_pd[-1], f"{err_pd[-1]:.2f}", PLOT_COLORS['pd'])
    _annotate_series_end(ax, t[-1], err_adrc[-1], f"{err_adrc[-1]:.2f}", PLOT_COLORS['adrc'])
    _metric_box(
        ax,
        [
            f'PD: Ts={pd_settle:.2f}s, Final={err_pd[-1]:.2f}deg',
            f'ADRC: Ts={adrc_settle:.2f}s, Final={err_adrc[-1]:.2f}deg',
        ],
        loc='upper right',
    )
    ax.legend(loc='upper right')

    ax = axes[0, 1]
    w_pd_norm = np.rad2deg(np.linalg.norm(results_pd['w'], axis=1))
    w_adrc_norm = np.rad2deg(np.linalg.norm(results_adrc['w'], axis=1))
    ax.fill_between(t, w_pd_norm, color=PLOT_COLORS['pd'], alpha=0.06)
    ax.fill_between(t, w_adrc_norm, color=PLOT_COLORS['adrc'], alpha=0.06)
    ax.plot(t, w_pd_norm, color=PLOT_COLORS['pd'], linewidth=2.2, label='PD')
    ax.plot(t, w_adrc_norm, color=PLOT_COLORS['adrc'], linewidth=2.2, linestyle='--', label='ADRC')
    _style_axes(ax, xlabel='时间 (s)', ylabel='角速度幅值 (deg/s)', title='角速度对比')
    _apply_time_axis(ax, t)
    _panel_label(ax, '(b)')
    _metric_box(
        ax,
        [
            f'PD 峰值 = {np.max(w_pd_norm):.2f}',
            f'ADRC 峰值 = {np.max(w_adrc_norm):.2f}',
        ],
        loc='upper right',
    )
    ax.legend(loc='upper right')

    ax = axes[1, 0]
    u_pd_norm = np.linalg.norm(results_pd['u'], axis=1)
    u_adrc_norm = np.linalg.norm(results_adrc['u'], axis=1)
    ax.fill_between(t, u_pd_norm, color=PLOT_COLORS['pd'], alpha=0.06)
    ax.fill_between(t, u_adrc_norm, color=PLOT_COLORS['adrc'], alpha=0.06)
    ax.plot(t, u_pd_norm, color=PLOT_COLORS['pd'], linewidth=2.2, label='PD')
    ax.plot(t, u_adrc_norm, color=PLOT_COLORS['adrc'], linewidth=2.2, linestyle='--', label='ADRC')
    ax.axhline(y=umax, color=PLOT_COLORS['limit'], linestyle=':', alpha=0.9, label=f'限制 {umax}')
    _style_axes(ax, xlabel='时间 (s)', ylabel='||u|| (N·m)', title='控制力矩对比')
    _apply_time_axis(ax, t)
    _panel_label(ax, '(c)')
    _metric_box(
        ax,
        [
            f'PD 能耗 = {results_pd["effort"]:.3f}',
            f'ADRC 能耗 = {results_adrc["effort"]:.3f}',
        ],
        loc='upper right',
    )
    ax.legend(loc='upper right')

    ax = axes[1, 1]
    metrics = ['ITAE\n(deg·s^2)', 'ISU\n(N^2·m^2·s)', '调节时间\n(s)', '超调量\n(度)']
    pd_values_raw = np.array([
        results_pd['ITAE'],
        results_pd['ISU'],
        results_pd['settle_time'],
        results_pd['overshoot'],
    ], dtype=float)
    adrc_values_raw = np.array([
        results_adrc['ITAE'],
        results_adrc['ISU'],
        results_adrc['settle_time'],
        results_adrc['overshoot'],
    ], dtype=float)
    denom = np.maximum(np.maximum(pd_values_raw, adrc_values_raw), 1e-12)
    x = np.arange(len(metrics))
    width = 0.36
    pd_norm = pd_values_raw / denom
    adrc_norm = adrc_values_raw / denom
    bars_pd = ax.bar(x - width / 2, pd_norm, width, label='PD', color=PLOT_COLORS['pd'], alpha=0.88)
    bars_adrc = ax.bar(x + width / 2, adrc_norm, width, label='ADRC', color=PLOT_COLORS['adrc'], alpha=0.84)
    for i in range(len(metrics)):
        ax.text(x[i] - width / 2, min(pd_norm[i] + 0.04, 1.14), f'{pd_values_raw[i]:.2f}',
                ha='center', va='bottom', fontsize=8, color=PLOT_COLORS['pd'])
        ax.text(x[i] + width / 2, min(adrc_norm[i] + 0.04, 1.14), f'{adrc_values_raw[i]:.2f}',
                ha='center', va='bottom', fontsize=8, color=PLOT_COLORS['adrc'])
        ax.plot([x[i] - width / 2, x[i] + width / 2], [max(pd_norm[i], adrc_norm[i]) + 0.015] * 2,
                color='#7f8b97', linewidth=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0, 1.18)
    _style_axes(ax, ylabel='归一化数值 (0~1)', title='性能指标对比')
    _panel_label(ax, '(d)')
    ax.grid(True, axis='y', color=PLOT_COLORS['grid'], linewidth=0.8, alpha=0.8)
    itae_impr = (results_pd['ITAE'] - results_adrc['ITAE']) / max(results_pd['ITAE'], 1e-12) * 100.0
    isu_impr = (results_pd['ISU'] - results_adrc['ISU']) / max(results_pd['ISU'], 1e-12) * 100.0
    settle_impr = (results_pd['settle_time'] - results_adrc['settle_time']) / max(results_pd['settle_time'], 1e-12) * 100.0
    _metric_box(
        ax,
        [
            f'ITAE 改进 = {itae_impr:+.1f}%',
            f'ISU 改进 = {isu_impr:+.1f}%',
            f'调节时间改进 = {settle_impr:+.1f}%',
        ],
        loc='upper left',
    )
    ax.legend(loc='upper right')

    return _finish_figure(fig, title)


def plot_simulation_process_overview(results, title="闭环仿真过程总览", error_ref_deg=1.0):
    """
    绘制更适合论文/汇报的单次仿真过程图。
    """
    t = np.asarray(results['t'], dtype=float)
    err = np.asarray(results['err'], dtype=float)
    w = np.asarray(results['w'], dtype=float)
    u = np.asarray(results['u'], dtype=float)
    main_color = _controller_color(results.get('controller_type', 'PD'))
    bias_true = np.asarray(results.get('gyro_bias_true'), dtype=float)
    bias_est = np.asarray(results.get('gyro_bias_est'), dtype=float)
    settle_time = float(results.get('settle_time', t[-1]))
    umax = float(results.get('u_limit', np.max(np.linalg.norm(u, axis=1)) + 1e-12))

    w_norm = np.rad2deg(np.linalg.norm(w, axis=1))
    u_norm = np.linalg.norm(u, axis=1)
    bias_true_norm = np.linalg.norm(bias_true, axis=1)
    bias_est_norm = np.linalg.norm(bias_est, axis=1)
    sat_mask = u_norm >= 0.98 * umax

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.2), sharex='col')

    ax = axes[0, 0]
    ax.plot(t, err, color=main_color, linewidth=2.25, label='姿态误差')
    ax.fill_between(t, err, color=main_color, alpha=0.10)
    ax.axhline(error_ref_deg, color=PLOT_COLORS['accent'], linestyle='--', linewidth=1.2, label=f'{error_ref_deg}度阈值')
    if settle_time <= t[-1]:
        ax.axvline(settle_time, color=PLOT_COLORS['limit'], linestyle=':', linewidth=1.3, label=f'调节时间 {settle_time:.2f}s')
        ax.axvspan(settle_time, t[-1], color='#d8f3dc', alpha=0.35)
    _style_axes(ax, ylabel='姿态误差 (deg)', title='姿态误差收敛过程')
    _apply_time_axis(ax, t)
    _panel_label(ax, '(a)')
    _annotate_series_end(ax, t[-1], err[-1], f"{err[-1]:.2f}", main_color)
    _metric_box(ax, [f'调节时间 = {settle_time:.2f} s', f'末端误差 = {err[-1]:.2f} deg'], loc='upper right')
    ax.legend(loc='upper right')

    ax = axes[0, 1]
    ax.plot(t, w_norm, color=PLOT_COLORS['z'], linewidth=2.1, label='角速度范数')
    _style_axes(ax, ylabel='||ω|| (deg/s)', title='角速度衰减过程')
    _apply_time_axis(ax, t)
    _panel_label(ax, '(b)')
    _annotate_series_end(ax, t[-1], w_norm[-1], f"{w_norm[-1]:.2f}", PLOT_COLORS['z'])
    _metric_box(ax, [f'峰值 = {np.max(w_norm):.2f} deg/s', f'末端 = {w_norm[-1]:.2f} deg/s'], loc='upper right')
    ax.legend(loc='upper right')

    ax = axes[1, 0]
    ax.plot(t, u_norm, color=PLOT_COLORS['truth'], linewidth=2.2, label='控制力矩范数')
    ax.axhline(umax, color=PLOT_COLORS['limit'], linestyle='--', linewidth=1.1, label=f'力矩上限 {umax:.2f}')
    if np.any(sat_mask):
            ax.fill_between(t, 0.0, u_norm, where=sat_mask, color=PLOT_COLORS['adrc'], alpha=0.16, label='近饱和区')
    _style_axes(ax, xlabel='时间 (s)', ylabel='||u|| (N·m)', title='控制输入与饱和过程')
    _apply_time_axis(ax, t)
    _panel_label(ax, '(c)')
    _annotate_series_end(ax, t[-1], u_norm[-1], f"{u_norm[-1]:.3f}", PLOT_COLORS['truth'])
    _metric_box(ax, [f'控制能耗 = {_curve_integral(u_norm, t):.3f} N·m·s'], loc='upper right')
    ax.legend(loc='upper right')

    ax = axes[1, 1]
    ax.plot(t, bias_true_norm, color=PLOT_COLORS['truth'], linewidth=2.0, label='真实偏置范数')
    ax.plot(t, bias_est_norm, color=PLOT_COLORS['estimate'], linestyle='--', linewidth=2.0, label='估计偏置范数')
    _style_axes(ax, xlabel='时间 (s)', ylabel='||b|| (rad/s)', title='陀螺偏置估计过程')
    _apply_time_axis(ax, t)
    _panel_label(ax, '(d)')
    _annotate_series_end(ax, t[-1], bias_est_norm[-1], f"{bias_est_norm[-1]:.4f}", PLOT_COLORS['estimate'])
    _metric_box(ax, [f'偏置估计误差 = {abs(bias_true_norm[-1] - bias_est_norm[-1]):.4f}'], loc='upper right')
    ax.legend(loc='upper right')

    return _finish_figure(fig, title)


def plot_inertia_identification(results, title="在线惯量辨识过程"):
    """
    绘制对角惯量在线辨识演化图。
    """
    t = np.asarray(results['t'], dtype=float)
    inertia_est = np.asarray(results.get('inertia_est'), dtype=float)
    if inertia_est.ndim != 2 or inertia_est.shape[1] != 3:
        raise ValueError("results['inertia_est'] 必须是 N×3 数组")
    inertia_ref = results.get('inertia_reference')
    if inertia_ref is None:
        inertia_ref = inertia_est[0]
    inertia_ref = np.asarray(inertia_ref, dtype=float).reshape(3)
    inertia_err = inertia_est - inertia_ref[None, :]
    update_mask = np.asarray(results.get('inertia_update_mask', np.ones_like(t)), dtype=float).reshape(-1)
    wdot_est = np.asarray(results.get('wdot_est', np.zeros((len(t), 3))), dtype=float)
    excitation = np.linalg.norm(wdot_est, axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(12.2, 10.2), sharex=True)
    labels = [('Jxx', PLOT_COLORS['x']), ('Jyy', PLOT_COLORS['y']), ('Jzz', PLOT_COLORS['z'])]

    ax = axes[0]
    for idx, (label, color) in enumerate(labels):
        ax.plot(t, inertia_est[:, idx], color=color, linewidth=2.1, label=f'{label}估计')
        ax.axhline(inertia_ref[idx], color=color, linestyle='--', linewidth=1.1, alpha=0.65, label=f'{label}参考' if idx == 0 else None)
        _annotate_series_end(ax, t[-1], inertia_est[-1, idx], f"{inertia_est[-1, idx]:.4f}", color)
    _style_axes(ax, ylabel='惯量对角元 (kg·m^2)', title=f"{title} | 估计轨迹")
    _apply_time_axis(ax, t)
    _panel_label(ax, '(a)')
    _metric_box(
        ax,
        [
            f'参考值 = [{inertia_ref[0]:.4f}, {inertia_ref[1]:.4f}, {inertia_ref[2]:.4f}]',
            f'末值 = [{inertia_est[-1, 0]:.4f}, {inertia_est[-1, 1]:.4f}, {inertia_est[-1, 2]:.4f}]',
        ],
        loc='lower right',
    )
    ax.legend(loc='upper right', ncol=2)

    ax = axes[1]
    for idx, (label, color) in enumerate(labels):
        ax.plot(t, inertia_err[:, idx], color=color, linewidth=1.9, label=f'{label}误差')
        ax.axhline(0.0, color='#9aa5b1', linestyle=':', linewidth=0.9)
    err_norm = np.linalg.norm(inertia_err, axis=1)
    ax.fill_between(t, 0.0, err_norm, color=PLOT_COLORS['soft_fill'], alpha=0.55, label='误差范数底纹')
    _style_axes(ax, ylabel='估计误差 (kg·m^2)', title='参考惯量偏差')
    _apply_time_axis(ax, t)
    _panel_label(ax, '(b)')
    _metric_box(
        ax,
        [
            f'最终误差范数 = {err_norm[-1]:.4e}',
            f'均方根误差 = {np.sqrt(np.mean(err_norm ** 2)):.4e}',
        ],
        loc='upper right',
    )
    ax.legend(loc='upper right', ncol=2)

    ax = axes[2]
    ax.plot(t, excitation, color=PLOT_COLORS['truth'], linewidth=2.0, label='||dot(omega)_est||')
    if update_mask.size == t.size and np.any(update_mask > 0.5):
        ax.fill_between(
            t,
            0.0,
            np.maximum(excitation, 1e-12),
            where=update_mask > 0.5,
            color=PLOT_COLORS['success'],
            alpha=0.20,
            label='参数更新活跃区',
        )
    _style_axes(ax, xlabel='时间 (s)', ylabel='激励强度', title='辨识激励与更新活跃区间')
    _apply_time_axis(ax, t)
    _panel_label(ax, '(c)')
    update_ratio = 100.0 * float(np.mean(update_mask > 0.5)) if update_mask.size == t.size else 0.0
    _metric_box(
        ax,
        [
            f'平均激励 = {np.mean(excitation):.4e}',
            f'更新占比 = {update_ratio:.1f}%',
        ],
        loc='upper right',
    )
    ax.legend(loc='upper right')

    return _finish_figure(fig, title)


def plot_inertia_identification_dashboard(results, title="动力学参数辨识综合图"):
    """
    绘制适合汇报/论文的动力学参数辨识综合仪表板。
    """
    t = np.asarray(results['t'], dtype=float)
    inertia_est = np.asarray(results.get('inertia_est'), dtype=float)
    inertia_ref = results.get('inertia_reference')
    if inertia_ref is None:
        inertia_ref = inertia_est[0]
    inertia_ref = np.asarray(inertia_ref, dtype=float).reshape(3)
    inertia_err = inertia_est - inertia_ref[None, :]
    wdot_est = np.asarray(results.get('wdot_est', np.zeros((len(t), 3))), dtype=float)
    u_hist = np.asarray(results.get('u', np.zeros((len(t), 3))), dtype=float)
    update_mask = np.asarray(results.get('inertia_update_mask', np.ones_like(t)), dtype=float).reshape(-1)
    regressor_min_sv = np.asarray(results.get('regressor_min_sv', np.full(len(t), np.nan)), dtype=float).reshape(-1)
    regressor_min_sv_ref = results.get('regressor_min_sv_reference')
    scheme = str(results.get('inertia_estimator_scheme', 'Unknown'))

    fig, axes = plt.subplots(3, 2, figsize=(14.8, 14.2), sharex='col')
    labels = [('Jxx', PLOT_COLORS['x']), ('Jyy', PLOT_COLORS['y']), ('Jzz', PLOT_COLORS['z'])]
    update_ratio = 100.0 * float(np.mean(update_mask > 0.5)) if update_mask.size == t.size else 0.0
    err_norm = np.linalg.norm(inertia_err, axis=1)

    ax = axes[0, 0]
    _highlight_active_spans(ax, t, update_mask, PLOT_COLORS['success'], alpha=0.06)
    for idx, (label, color) in enumerate(labels):
        ax.plot(t, inertia_est[:, idx], color=color, linewidth=2.15, label=f'{label}估计')
        ax.axhline(inertia_ref[idx], color=color, linestyle='--', linewidth=1.0, alpha=0.65)
        _annotate_series_end(ax, t[-1], inertia_est[-1, idx], f"{inertia_est[-1, idx]:.4f}", color)
    _style_axes(ax, ylabel='惯量 (kg·m^2)', title=f'辨识方案: {scheme}')
    _panel_label(ax, '(a)')
    _apply_time_axis(ax, t)
    _metric_box(
        ax,
        [f'参考惯量 = [{inertia_ref[0]:.3f}, {inertia_ref[1]:.3f}, {inertia_ref[2]:.3f}]'],
        loc='lower right',
    )
    ax.legend(loc='upper right', ncol=2)

    ax = axes[0, 1]
    _highlight_active_spans(ax, t, update_mask, PLOT_COLORS['success'], alpha=0.05)
    for idx, (label, color) in enumerate(labels):
        ax.plot(t, inertia_err[:, idx], color=color, linewidth=1.7, label=f'{label}误差')
    ax.fill_between(t, err_norm, color=PLOT_COLORS['soft_fill'], alpha=0.45, zorder=0)
    ax.plot(t, err_norm, color=PLOT_COLORS['truth'], linewidth=2.15, linestyle='-.', label='误差范数')
    ax.axhline(0.0, color='#9aa5b1', linestyle=':', linewidth=0.9)
    _style_axes(ax, ylabel='误差 (kg·m^2)', title='辨识误差收敛')
    _panel_label(ax, '(b)')
    _apply_time_axis(ax, t)
    _metric_box(ax, [f'最终误差范数 = {err_norm[-1]:.4e}', f'RMSE = {np.sqrt(np.mean(err_norm ** 2)):.4e}'])
    ax.legend(loc='upper right', ncol=2)

    ax = axes[1, 0]
    torque_norm = np.linalg.norm(u_hist, axis=1)
    accel_norm = np.linalg.norm(wdot_est, axis=1)
    _highlight_active_spans(ax, t, update_mask, PLOT_COLORS['success'], alpha=0.05)
    ax.fill_between(t, torque_norm, color=PLOT_COLORS['pd'], alpha=0.05)
    ax.fill_between(t, accel_norm, color=PLOT_COLORS['adrc'], alpha=0.05)
    ax.plot(t, torque_norm, color=PLOT_COLORS['pd'], linewidth=2.0, label='||u||')
    ax.plot(t, accel_norm, color=PLOT_COLORS['adrc'], linewidth=2.0, label='||dot(omega)_est||')
    _style_axes(ax, xlabel='时间 (s)', ylabel='激励量级', title='辨识所需激励条件')
    _panel_label(ax, '(c)')
    _apply_time_axis(ax, t)
    _metric_box(
        ax,
        [
            f'平均 ||u|| = {np.mean(torque_norm):.3f}',
            f'平均 ||dot(omega)|| = {np.mean(accel_norm):.3f}',
        ],
        loc='upper right',
    )
    ax.legend(loc='upper right')

    ax = axes[1, 1]
    finite_mask = np.isfinite(regressor_min_sv)
    _highlight_active_spans(ax, t, update_mask, PLOT_COLORS['success'], alpha=0.05)
    if np.any(finite_mask):
        ax.plot(t[finite_mask], regressor_min_sv[finite_mask], color=PLOT_COLORS['accent'], linewidth=2.0, label=r'$\sigma_{\min}(\Phi)$')
        ax.fill_between(t[finite_mask], regressor_min_sv[finite_mask], color=PLOT_COLORS['accent'], alpha=0.12)
    else:
        ax.plot(t, np.zeros_like(t), color=PLOT_COLORS['limit'], linewidth=1.2, linestyle='--', label='PE指标不可用')
    if regressor_min_sv_ref is not None and np.isfinite(regressor_min_sv_ref):
        ax.axhline(
            float(regressor_min_sv_ref),
            color=PLOT_COLORS['limit'],
            linestyle='--',
            linewidth=1.1,
            label=f'参考阈值 {float(regressor_min_sv_ref):.3f}',
        )
    _style_axes(ax, ylabel=r'$\sigma_{\min}(\Phi)$', title='持续激励（PE）诊断')
    _panel_label(ax, '(d)')
    _apply_time_axis(ax, t)
    if np.any(finite_mask):
        _metric_box(
            ax,
            [
                f'均值 = {np.nanmean(regressor_min_sv):.4e}',
                f'最小值 = {np.nanmin(regressor_min_sv):.4e}',
            ],
            loc='upper right',
        )
    ax.legend(loc='upper right')

    ax = axes[2, 0]
    update_binary = (update_mask > 0.5).astype(float)
    dt_local = float(np.median(np.diff(t))) if t.size > 1 else 1.0
    smooth_window = max(5, int(round(0.75 / max(dt_local, 1e-9))))
    kernel = np.ones(smooth_window, dtype=float) / smooth_window
    activity_ratio = 100.0 * np.convolve(update_binary, kernel, mode='same')
    event_times = t[update_binary > 0.5]

    ax.fill_between(t, 0.0, activity_ratio, color=PLOT_COLORS['success'], alpha=0.18, label='局部更新率')
    ax.plot(t, activity_ratio, color=PLOT_COLORS['success'], linewidth=2.0)
    if event_times.size > 0:
        event_height = max(6.0, 0.12 * np.max(activity_ratio) + 4.0)
        ax.vlines(event_times, 0.0, event_height, color=PLOT_COLORS['accent'], linewidth=1.1, alpha=0.55, label='更新事件')
        active_span = f'{event_times[0]:.2f}s - {event_times[-1]:.2f}s'
    else:
        active_span = '无有效更新'

    _style_axes(ax, xlabel='时间 (s)', ylabel='局部更新率 (%)', title='参数更新活跃度时间线')
    _panel_label(ax, '(e)')
    _apply_time_axis(ax, t)
    ax.set_ylim(0.0, max(12.0, np.max(activity_ratio) * 1.20 + 2.0))
    _metric_box(
        ax,
        [
            f'更新占比 = {update_ratio:.1f}%',
            f'活跃区间 = {active_span}',
        ],
    )
    ax.legend(loc='upper right')

    ax = axes[2, 1]
    if np.any(finite_mask):
        active_mask = finite_mask & (update_mask > 0.5)
        sv_all = regressor_min_sv[finite_mask]
        err_all = err_norm[finite_mask]
        sv_active = regressor_min_sv[active_mask]
        err_active = err_norm[active_mask]

        x_hi = float(np.nanquantile(sv_all, 0.995))
        x_hi = max(x_hi, float(np.nanmax(sv_all)) * 0.75, 1e-4)
        x_lo = 0.0

        ax.scatter(
            sv_all,
            err_all,
            s=20,
            color=PLOT_COLORS['pd'],
            alpha=0.22,
            edgecolors='none',
            label='全时段样本',
        )
        if sv_active.size > 0:
            ax.scatter(
                sv_active,
                err_active,
                s=34,
                color=PLOT_COLORS['success'],
                alpha=0.62,
                edgecolors='white',
                linewidth=0.4,
                label='更新活跃样本',
            )
            bin_edges = np.linspace(max(x_lo, np.min(sv_active)), max(x_hi, np.max(sv_active)), num=7)
            trend_x = []
            trend_y = []
            for left, right in zip(bin_edges[:-1], bin_edges[1:]):
                bin_mask = (sv_active >= left) & (sv_active <= right if right == bin_edges[-1] else sv_active < right)
                if np.any(bin_mask):
                    trend_x.append(float(np.median(sv_active[bin_mask])))
                    trend_y.append(float(np.median(err_active[bin_mask])))
            if len(trend_x) >= 2:
                ax.plot(trend_x, trend_y, color=PLOT_COLORS['accent'], linewidth=2.0, marker='o', markersize=4, label='活跃样本趋势')
        ax.set_xlim(x_lo, x_hi * 1.05)
        corr_val = float(np.corrcoef(sv_all, err_all)[0, 1]) if sv_all.size >= 2 and np.std(sv_all) > 1e-12 and np.std(err_all) > 1e-12 else np.nan
    _style_axes(ax, xlabel=r'$\sigma_{\min}(\Phi)$', ylabel='误差范数 (kg·m^2)', title='PE 强度与辨识误差关联')
    _panel_label(ax, '(f)')
    if np.any(finite_mask):
        info_lines = [f'样本数 = {int(np.sum(finite_mask))}']
        if np.isfinite(corr_val):
            info_lines.append(f'相关系数 = {corr_val:+.2f}')
        _metric_box(ax, info_lines, loc='upper left')
        ax.legend(loc='upper right')

    return _finish_figure(fig, title)


# ==================== 增益调优相关绘图函数 ====================

def plot_gain_landscape(objective_fun, Kp_range, Kd_range, N=15, title="增益参数空间目标函数热图"):
    """
    绘制增益参数空间的目标函数热图
    
    参数:
        objective_fun: 目标函数，接受 [Kp, Kd] 返回标量
        Kp_range: (Kp_min, Kp_max)
        Kd_range: (Kd_min, Kd_max)
        N: 网格分辨率
        title: 图标题
    """
    Kp_vals = np.linspace(Kp_range[0], Kp_range[1], N)
    Kd_vals = np.linspace(Kd_range[0], Kd_range[1], N)
    
    Z = np.zeros((N, N))
    for i, Kp in enumerate(Kp_vals):
        for j, Kd in enumerate(Kd_vals):
            Z[j, i] = objective_fun(np.array([Kp, Kd]))
    
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    im = ax.contourf(Kp_vals, Kd_vals, Z, levels=20, cmap='viridis')
    cbar = plt.colorbar(im, ax=ax, label='目标函数值')
    ax.contour(Kp_vals, Kd_vals, Z, levels=10, colors='white', alpha=0.3, linewidths=0.5)
    ax.set_xlabel('Kp (比例增益)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Kd (微分增益)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    return fig


def plot_gain_landscape_from_cache(cache_dict, Kp_range, Kd_range, title="参数优化景观（基于已评估点）"):
    """
    基于缓存数据绘制增益参数空间的目标函数散点图和等高线
    
    参数:
        cache_dict: 缓存字典，key为(Kp, Kd)，value为{'score': ...}
        Kp_range: (Kp_min, Kp_max)
        Kd_range: (Kd_min, Kd_max)
        title: 图标题
    """
    if not cache_dict:
        print("[警告] 缓存为空，跳过景观图绘制")
        return None
    
    # 提取缓存中的点
    Kp_list = []
    Kd_list = []
    score_list = []
    
    for (Kp, Kd), value in cache_dict.items():
        if Kp_range[0] <= Kp <= Kp_range[1] and Kd_range[0] <= Kd <= Kd_range[1]:
            Kp_list.append(Kp)
            Kd_list.append(Kd)
            score_list.append(value.get('score', np.inf))
    
    if len(Kp_list) < 3:
        print(f"[警告] 缓存点数太少({len(Kp_list)})，跳过景观图绘制")
        return None
    
    Kp_arr = np.array(Kp_list)
    Kd_arr = np.array(Kd_list)
    score_arr = np.array(score_list)
    
    # 绘制散点和等高线
    fig, ax = plt.subplots(figsize=(11.5, 8.0))

    # 绘制基于已评估点的目标函数景观，更适合展示优化轨迹分布
    from matplotlib.tri import Triangulation
    if len(Kp_list) >= 4:
        try:
            tri = Triangulation(Kp_arr, Kd_arr)
            filled = ax.tricontourf(tri, score_arr, levels=18, cmap='viridis', alpha=0.88)
            ax.tricontour(tri, score_arr, levels=10, colors='white', alpha=0.35, linewidths=0.7)
            cbar = plt.colorbar(filled, ax=ax, label='风险敏感目标函数值（越小越好）')
        except Exception:
            scatter = ax.scatter(
                Kp_arr,
                Kd_arr,
                c=score_arr,
                cmap='viridis',
                s=95,
                edgecolors='black',
                linewidth=0.5,
                alpha=0.75,
                zorder=5,
            )
            cbar = plt.colorbar(scatter, ax=ax, label='风险敏感目标函数值（越小越好）')
    else:
        scatter = ax.scatter(
            Kp_arr,
            Kd_arr,
            c=score_arr,
            cmap='viridis',
            s=95,
            edgecolors='black',
            linewidth=0.5,
            alpha=0.75,
            zorder=5,
        )
        cbar = plt.colorbar(scatter, ax=ax, label='风险敏感目标函数值（越小越好）')

    ax.scatter(
        Kp_arr,
        Kd_arr,
        s=24,
        color='white',
        edgecolors='black',
        linewidth=0.45,
        alpha=0.55,
        zorder=6,
        label='已评估样本',
    )

    # 标记最优点
    best_idx = np.argmin(score_arr)
    best_kp = Kp_arr[best_idx]
    best_kd = Kd_arr[best_idx]
    best_score = score_arr[best_idx]
    ax.scatter(
        [best_kp],
        [best_kd],
        marker='*',
        s=520,
        color='#d1495b',
        edgecolors='#7f1d1d',
        linewidth=1.8,
        zorder=10,
        label=f'最优点 ({best_kp:.3f}, {best_kd:.3f})',
    )
    ax.annotate(
        f'Best score = {best_score:.3f}',
        xy=(best_kp, best_kd),
        xytext=(12, -22),
        textcoords='offset points',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=PLOT_COLORS['adrc'], alpha=0.92),
        arrowprops=dict(arrowstyle='->', color=PLOT_COLORS['adrc'], lw=1.0),
    )

    _style_axes(ax, xlabel='Kp (比例增益)', ylabel='Kd (微分增益)', title=title)
    ax.set_xlim(Kp_range)
    ax.set_ylim(Kd_range)
    ax.legend(loc='upper right')
    return _finish_figure(fig)


def plot_optimizer_score_distribution(method_run_data, title="优化器目标值分布对比"):
    """
    绘制各优化器在重复试验中的目标函数分布，用于展示稳健性。
    """
    methods = list(method_run_data.keys())
    if len(methods) == 0:
        raise ValueError("method_run_data 不能为空")

    score_sets = []
    for method in methods:
        runs = method_run_data[method].get('runs', [])
        scores = np.asarray([r['score'] for r in runs], dtype=float)
        score_sets.append(scores if scores.size > 0 else np.array([np.nan], dtype=float))

    fig, ax = plt.subplots(figsize=(11.8, 6.5))
    positions = np.arange(1, len(methods) + 1)
    cmap = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    violin = ax.violinplot(score_sets, positions=positions, widths=0.8, showmeans=False, showextrema=False, showmedians=False)
    for body, color in zip(violin['bodies'], cmap):
        body.set_facecolor(color)
        body.set_edgecolor('#2f3b4a')
        body.set_alpha(0.45)

    for pos, method, scores, color in zip(positions, methods, score_sets, cmap):
        clean_scores = scores[np.isfinite(scores)]
        if clean_scores.size == 0:
            continue
        ax.boxplot(
            clean_scores,
            positions=[pos],
            widths=0.22,
            patch_artist=True,
            boxprops=dict(facecolor='white', edgecolor='#2f3b4a', linewidth=1.2),
            medianprops=dict(color='#1f1f1f', linewidth=1.5),
            whiskerprops=dict(color='#2f3b4a', linewidth=1.0),
            capprops=dict(color='#2f3b4a', linewidth=1.0),
            flierprops=dict(marker='o', markersize=4, markerfacecolor=color, markeredgecolor='#2f3b4a', alpha=0.55),
        )
        rng = np.random.RandomState(100 + pos)
        jitter = pos + 0.09 * (rng.rand(clean_scores.size) - 0.5)
        ax.scatter(jitter, clean_scores, s=28, color=color, edgecolors='white', linewidth=0.5, alpha=0.9, zorder=6)
        ax.text(
            pos,
            np.max(clean_scores) + 0.015 * max(np.ptp(clean_scores), 1.0),
            f'μ={np.mean(clean_scores):.3f}\nσ={np.std(clean_scores):.3f}',
            ha='center',
            va='bottom',
            fontsize=8.5,
            color='#2f3b4a',
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(methods, rotation=10)
    _style_axes(ax, ylabel='风险敏感目标函数值', title=title)
    return _finish_figure(fig)


def plot_optimizer_parameter_scatter(method_run_data, bounds, title="优化器参数收敛区域分布"):
    """
    绘制各优化器多次运行得到的 (Kp, Kd) 参数分布，便于观察收敛簇。
    """
    methods = list(method_run_data.keys())
    if len(methods) == 0:
        raise ValueError("method_run_data 不能为空")

    lo = np.asarray(bounds[0], dtype=float)
    hi = np.asarray(bounds[1], dtype=float)

    fig, ax = plt.subplots(figsize=(10.8, 7.4))
    cmap = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    for method, color in zip(methods, cmap):
        runs = method_run_data[method].get('runs', [])
        if len(runs) == 0:
            continue
        kp_vals = np.asarray([r['Kp'] for r in runs], dtype=float)
        kd_vals = np.asarray([r['Kd'] for r in runs], dtype=float)
        score_vals = np.asarray([r['score'] for r in runs], dtype=float)

        size = 90.0 + 210.0 * (np.max(score_vals) - score_vals) / max(np.ptp(score_vals), 1e-9)
        ax.scatter(
            kp_vals,
            kd_vals,
            s=size,
            color=color,
            alpha=0.68,
            edgecolors='white',
            linewidth=0.8,
            label=method,
        )

        best_idx = int(np.argmin(score_vals))
        ax.scatter(
            [kp_vals[best_idx]],
            [kd_vals[best_idx]],
            marker='*',
            s=340,
            color=color,
            edgecolors='#2f3b4a',
            linewidth=1.2,
            zorder=8,
        )
        ax.annotate(
            method,
            xy=(kp_vals[best_idx], kd_vals[best_idx]),
            xytext=(7, 6),
            textcoords='offset points',
            fontsize=9,
            color='#2f3b4a',
        )

    _style_axes(ax, xlabel='Kp (比例增益)', ylabel='Kd (微分增益)', title=title)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.legend(loc='upper right', ncol=2)
    return _finish_figure(fig)


def plot_optimizer_convergence(comparison_results, title="优化器收敛曲线对比"):
    """
    绘制多个优化器的收敛曲线对比
    
    参数:
        comparison_results: 列表，每个元素为 {'method': '算法名', 'history': 收敛历史数组}
        title: 图标题
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    
    for result in comparison_results:
        if 'history' in result and result['history'] is not None:
            hist = np.array(result['history'], dtype=float)
            if hist.size > 0:
                ax.plot(hist, linewidth=2, marker='o', markersize=3, 
                       label=f"{result['method']}", alpha=0.8)
    
    ax.set_xlabel('迭代步', fontsize=12, fontweight='bold')
    ax.set_ylabel('最优目标函数值', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_gain_comparison_bar(comparison_results, metric='score', title="优化结果对比"):
    """
    绘制不同优化算法的性能对比（柱状图）
    
    参数:
        comparison_results: 列表，每个元素为 {'method': '算法名', 'score': 目标函数值, ...}
        metric: 要对比的指标 ('score', 'settle_time', 'effort', 等)
        title: 图标题
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    
    methods = [r['method'] for r in comparison_results]
    values = [r.get(metric, r.get('score', 0)) for r in comparison_results]
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    bars = ax.bar(methods, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel(f'{metric}', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 在柱顶添加数值标签
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    return fig


def plot_pareto_front(comparison_results, title="Pareto前沿：稳定时间 vs 控制能量"):
    """
    绘制Pareto前沿（权衡曲线）
    
    参数:
        comparison_results: 列表，每个元素为{'method':'算法名', 'settle_time':..., 'effort':...}
        title: 图标题
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    methods = []
    settle_times = []
    efforts = []
    colors_list = plt.cm.Set3(np.linspace(0, 1, len(comparison_results)))
    
    for i, result in enumerate(comparison_results):
        methods.append(result['method'])
        settle_times.append(result.get('settle_time', 0))
        efforts.append(result.get('effort', 0))
        
        ax.scatter(settle_times[-1], efforts[-1], s=220, c=[colors_list[i]],
                  edgecolors='black', linewidth=2, alpha=0.8, zorder=5)
        ax.annotate(result['method'], 
                   (settle_times[-1], efforts[-1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, bbox=dict(boxstyle='round,pad=0.3', 
                                        facecolor='yellow', alpha=0.5))
    
    if len(settle_times) >= 2:
        pts = np.column_stack([settle_times, efforts])
        order = np.argsort(pts[:, 0])
        pts_sorted = pts[order]
        pareto_idx = []
        best_effort = np.inf
        for idx, (_, effort_val) in enumerate(pts_sorted):
            if effort_val <= best_effort + 1e-12:
                pareto_idx.append(idx)
                best_effort = effort_val
        pareto_pts = pts_sorted[pareto_idx]
        if pareto_pts.shape[0] >= 2:
            ax.plot(
                pareto_pts[:, 0],
                pareto_pts[:, 1],
                color=PLOT_COLORS['truth'],
                linewidth=1.6,
                linestyle='--',
                alpha=0.75,
                label='Pareto近似前沿',
            )

    _style_axes(ax, xlabel='稳定时间 (s)', ylabel='控制能量 ∫||u||dt (N·m·s)', title=title)
    ax.legend(loc='upper right')
    return _finish_figure(fig)


def plot_gain_metrics_heatmap(comparison_results, title="增益参数对性能指标的影响"):
    """
    绘制增益参数对多个性能指标的影响热图
    
    参数:
        comparison_results: 列表，每个元素包含各种性能指标
        title: 图标题
    """
    # 提取关键指标和算法名
    methods = [r['method'] for r in comparison_results]
    metrics_dict = {}
    
    metric_keys = ['settle_time', 'overshoot', 'final_error', 'effort', 'sat_ratio']
    metric_display = {
        'settle_time': 'Ts',
        'overshoot': 'Overshoot',
        'final_error': 'Final error',
        'effort': 'Effort',
        'sat_ratio': 'Sat. ratio',
    }
    for key in metric_keys:
        metrics_dict[key] = []
        for result in comparison_results:
            val = result.get(key, 0)
            if key == 'sat_ratio' and val is not None:
                val = val * 100  # 转换为百分比
            metrics_dict[key].append(float(val) if val else 0)
    
    # 标准化（0-1范围）
    for key in metrics_dict:
        vals = np.array(metrics_dict[key])
        if np.max(vals) > np.min(vals):
            metrics_dict[key] = (vals - np.min(vals)) / (np.max(vals) - np.min(vals))
        else:
            metrics_dict[key] = vals
    
    # 创建热图数据
    heatmap_data = []
    metric_labels = []
    for key in metric_keys:
        if key in metrics_dict:
            heatmap_data.append(metrics_dict[key])
            metric_labels.append(metric_display.get(key, key))
    
    heatmap_data = np.array(heatmap_data)
    
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto')
    
    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(metric_labels)))
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.set_yticklabels(metric_labels)
    
    # 在单元格中添加数值
    for i in range(len(metric_labels)):
        for j in range(len(methods)):
            text = ax.text(j, i, f'{heatmap_data[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('标准化指标值', rotation=270, labelpad=20)
    
    plt.tight_layout()
    return fig


def plot_multiple_responses(results_dict, title="不同增益参数的时域响应对比"):
    """
    绘制不同增益参数下的时域响应对比
    
    参数:
        results_dict: 字典，key为参数标签，value为{'t': 时间, 'err': 误差, 'u': 控制力矩, ...}
        title: 图标题
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(results_dict)))
    
    for idx, (label, result) in enumerate(results_dict.items()):
        t = result['t']
        err = result['err']
        w = result['w']
        u = result['u']
        
        # 姿态误差
        axes[0].plot(t, err, linewidth=2, label=label, color=colors[idx])
        
        # 角速度幅值
        w_norm = np.linalg.norm(w, axis=1)
        axes[1].plot(t, np.rad2deg(w_norm), linewidth=2, label=label, color=colors[idx])
        
        # 控制力矩幅值
        u_norm = np.linalg.norm(u, axis=1)
        axes[2].plot(t, u_norm, linewidth=2, label=label, color=colors[idx])
    
    _style_axes(axes[0], ylabel='姿态误差 (度)')
    _panel_label(axes[0], '(a)')
    axes[0].legend(loc='upper right')
    
    _style_axes(axes[1], ylabel='||ω|| (deg/s)')
    _panel_label(axes[1], '(b)')
    axes[1].legend(loc='upper right')
    
    _style_axes(axes[2], xlabel='时间 (s)', ylabel='||u|| (N·m)')
    _panel_label(axes[2], '(c)')
    axes[2].legend(loc='upper right')

    for ax in axes:
        _apply_time_axis(ax, t)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_phase_portrait(results_pd, results_adrc, title="相平面图对比：姿态误差-角速度"):
    """
    绘制 PD 与 ADRC 的相平面轨迹（姿态误差角 vs 角速度幅值）。

    参数:
        results_pd: PD 控制器仿真结果字典，需包含 'err' 与 'w'
        results_adrc: ADRC 控制器仿真结果字典，需包含 'err' 与 'w'
        title: 图标题
    """
    err_pd = np.asarray(results_pd['err'], dtype=float)
    err_adrc = np.asarray(results_adrc['err'], dtype=float)
    w_pd = np.asarray(results_pd['w'], dtype=float)
    w_adrc = np.asarray(results_adrc['w'], dtype=float)

    w_pd_norm = np.rad2deg(np.linalg.norm(w_pd, axis=1))
    w_adrc_norm = np.rad2deg(np.linalg.norm(w_adrc, axis=1))

    fig, ax = plt.subplots(figsize=(9.5, 7.2))

    # 主相轨迹
    ax.plot(err_pd, w_pd_norm, color=PLOT_COLORS['pd'], linewidth=2.2, label='PD相轨迹')
    ax.plot(err_adrc, w_adrc_norm, color=PLOT_COLORS['adrc'], linestyle='--', linewidth=2.2, label='ADRC相轨迹')

    # 起点（圆圈）
    ax.plot(err_pd[0], w_pd_norm[0], marker='o', markersize=8, color=PLOT_COLORS['pd'], markerfacecolor='none', label='PD起点')
    ax.plot(err_adrc[0], w_adrc_norm[0], marker='o', markersize=8, color=PLOT_COLORS['adrc'], markerfacecolor='none', label='ADRC起点')

    # 终点（星号）
    ax.plot(err_pd[-1], w_pd_norm[-1], marker='*', markersize=12, color=PLOT_COLORS['pd'], label='PD终点')
    ax.plot(err_adrc[-1], w_adrc_norm[-1], marker='*', markersize=12, color=PLOT_COLORS['adrc'], label='ADRC终点')

    _style_axes(ax, xlabel='姿态误差角 (deg)', ylabel='角速度幅值 (deg/s)', title=title)
    _metric_box(
        ax,
        [
            f'PD 末端: ({err_pd[-1]:.2f}, {w_pd_norm[-1]:.2f})',
            f'ADRC末端: ({err_adrc[-1]:.2f}, {w_adrc_norm[-1]:.2f})',
        ],
        loc='lower left',
    )
    ax.legend(loc='upper right', fontsize=10)
    return _finish_figure(fig)


def plot_simulation_report_dashboard(results, title="单次仿真综合仪表板"):
    """
    生成单次仿真的高信息密度总览图，适合论文附图或汇报封面图。
    """
    t = np.asarray(results['t'], dtype=float)
    err = np.asarray(results['err'], dtype=float)
    w = np.asarray(results['w'], dtype=float)
    u = np.asarray(results['u'], dtype=float)
    q_true = np.asarray(results['q_true'], dtype=float)
    q_est = np.asarray(results['q_est'], dtype=float)
    bias_true = np.asarray(results['gyro_bias_true'], dtype=float)
    bias_est = np.asarray(results['gyro_bias_est'], dtype=float)
    main_color = _controller_color(results.get('controller_type', 'PD'))

    q_err = quat_angle_errors_deg(q_true, q_est)
    w_norm = np.rad2deg(np.linalg.norm(w, axis=1))
    u_norm = np.linalg.norm(u, axis=1)
    bias_err = np.linalg.norm(bias_true - bias_est, axis=1)

    fig, axes = plt.subplots(3, 2, figsize=(15.2, 11.6), sharex='col')

    ax = axes[0, 0]
    ax.plot(t, err, color=main_color, linewidth=2.2)
    ax.fill_between(t, err, color=main_color, alpha=0.10)
    ax.axhline(1.0, color=PLOT_COLORS['accent'], linestyle='--', linewidth=1.0)
    _style_axes(ax, ylabel='姿态误差 (deg)', title='闭环姿态误差')
    _panel_label(ax, '(a)')
    _apply_time_axis(ax, t)
    _metric_box(ax, [f'Ts = {results["settle_time"]:.2f} s', f'Final = {results["final_error"]:.2f} deg'])

    ax = axes[0, 1]
    ax.plot(t, q_err, color=PLOT_COLORS['estimate'], linewidth=2.0)
    ax.fill_between(t, q_err, color=PLOT_COLORS['estimate'], alpha=0.10)
    _style_axes(ax, ylabel='估计误差 (deg)', title='姿态估计误差')
    _panel_label(ax, '(b)')
    _apply_time_axis(ax, t)
    _metric_box(ax, [f'RMSE = {np.sqrt(np.mean(q_err ** 2)):.2f} deg'])

    ax = axes[1, 0]
    ax.plot(t, w_norm, color=PLOT_COLORS['z'], linewidth=2.0)
    _style_axes(ax, ylabel='||ω|| (deg/s)', title='角速度衰减')
    _panel_label(ax, '(c)')
    _apply_time_axis(ax, t)
    _metric_box(ax, [f'Max = {np.max(w_norm):.2f}', f'Final = {w_norm[-1]:.2f}'])

    ax = axes[1, 1]
    ax.plot(t, u_norm, color=PLOT_COLORS['truth'], linewidth=2.0)
    ax.axhline(results.get('u_limit', np.max(u_norm)), color=PLOT_COLORS['limit'], linestyle='--', linewidth=1.0)
    _style_axes(ax, ylabel='||u|| (N·m)', title='控制输入范数')
    _panel_label(ax, '(d)')
    _apply_time_axis(ax, t)
    _metric_box(ax, [f'Effort = {results["effort"]:.3f}', f'Sat = {results["sat_ratio"]*100:.1f}%'])

    ax = axes[2, 0]
    ax.plot(t, np.linalg.norm(bias_true, axis=1), color=PLOT_COLORS['truth'], linewidth=1.8, label='真实偏置')
    ax.plot(t, np.linalg.norm(bias_est, axis=1), color=PLOT_COLORS['estimate'], linestyle='--', linewidth=1.8, label='估计偏置')
    _style_axes(ax, xlabel='时间 (s)', ylabel='||b|| (rad/s)', title='陀螺偏置跟踪')
    _panel_label(ax, '(e)')
    _apply_time_axis(ax, t)
    ax.legend(loc='upper right')

    ax = axes[2, 1]
    ax.plot(t, bias_err, color=PLOT_COLORS['adrc'], linewidth=2.0)
    ax.fill_between(t, bias_err, color=PLOT_COLORS['adrc'], alpha=0.10)
    _style_axes(ax, xlabel='时间 (s)', ylabel='偏置误差范数', title='估计误差收敛')
    _panel_label(ax, '(f)')
    _apply_time_axis(ax, t)
    _metric_box(ax, [f'Final = {bias_err[-1]:.4f}', f'Mean = {np.mean(bias_err):.4f}'])

    return _finish_figure(fig, title)


def plot_optimizer_report_dashboard(cache_dict, method_run_data, comparison_results, bounds, title="PD优化结果综合仪表板"):
    """
    生成优化结果综合仪表板，汇总景观、分布、收敛与参数分布。
    """
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 11.4))

    # (a) 参数景观
    ax = axes[0, 0]
    if cache_dict:
        kp_vals = []
        kd_vals = []
        score_vals = []
        for (kp, kd), rec in cache_dict.items():
            kp_vals.append(kp)
            kd_vals.append(kd)
            score_vals.append(rec.get('score', np.inf))
        kp_arr = np.asarray(kp_vals, dtype=float)
        kd_arr = np.asarray(kd_vals, dtype=float)
        score_arr = np.asarray(score_vals, dtype=float)
        if kp_arr.size >= 4:
            from matplotlib.tri import Triangulation
            try:
                tri = Triangulation(kp_arr, kd_arr)
                filled = ax.tricontourf(tri, score_arr, levels=18, cmap='cividis', alpha=0.94)
                contour = ax.tricontour(tri, score_arr, levels=8, colors='white', linewidths=0.55, alpha=0.35)
                ax.clabel(contour, inline=True, fontsize=7, fmt='%.1f')
                cbar = fig.colorbar(filled, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Objective', fontsize=9)
            except Exception:
                scatter = ax.scatter(kp_arr, kd_arr, c=score_arr, cmap='cividis', s=55, alpha=0.82)
                cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Objective', fontsize=9)
        else:
            scatter = ax.scatter(kp_arr, kd_arr, c=score_arr, cmap='cividis', s=55, alpha=0.82)
            cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Objective', fontsize=9)
        ax.scatter(kp_arr, kd_arr, color='white', s=9, alpha=0.28, linewidths=0)
        best_idx = int(np.argmin(score_arr))
        ax.scatter([kp_arr[best_idx]], [kd_arr[best_idx]], marker='*', s=320, color=PLOT_COLORS['adrc'], edgecolors='white', linewidth=1.0)
        _metric_box(
            ax,
            [
                f'已评估点 = {kp_arr.size}',
                f'最优 = ({kp_arr[best_idx]:.2f}, {kd_arr[best_idx]:.2f})',
            ],
            loc='upper left',
        )
    _style_axes(ax, xlabel='Kp', ylabel='Kd', title='参数景观')
    _panel_label(ax, '(a)')
    ax.set_xlim(bounds[0][0], bounds[1][0])
    ax.set_ylim(bounds[0][1], bounds[1][1])

    # (b) 收敛
    ax = axes[0, 1]
    best_method = comparison_results[0]['method'] if comparison_results else None
    for row in comparison_results:
        hist = np.asarray(row.get('history', []), dtype=float)
        if hist.size == 0:
            continue
        if row['method'] == best_method:
            ax.plot(hist, linewidth=2.7, color=PLOT_COLORS['adrc'], label=f"{row['method']} (Best)")
        else:
            ax.plot(hist, linewidth=1.65, alpha=0.85, label=row['method'])
    _style_axes(ax, xlabel='迭代步', ylabel='Best-so-far objective', title='优化器收敛曲线')
    _panel_label(ax, '(b)')
    if comparison_results:
        _metric_box(
            ax,
            [f'最佳算法 = {best_method}', f'最佳 score = {comparison_results[0]["score_best"]:.3f}'],
            loc='upper right',
        )
    ax.legend(loc='upper right', fontsize=8.8)

    # (c) 目标值分布
    ax = axes[1, 0]
    methods = list(method_run_data.keys())
    if methods:
        positions = np.arange(1, len(methods) + 1)
        data = [np.asarray([r['score'] for r in method_run_data[m].get('runs', [])], dtype=float) for m in methods]
        violin = ax.violinplot(data, positions=positions, widths=0.75, showmeans=False, showextrema=False)
        cmap = plt.cm.Set2(np.linspace(0, 1, len(methods)))
        for body, color in zip(violin['bodies'], cmap):
            body.set_facecolor(color)
            body.set_edgecolor('#2f3b4a')
            body.set_alpha(0.48)
        ax.boxplot(
            data,
            positions=positions,
            widths=0.20,
            patch_artist=True,
            showcaps=True,
            boxprops=dict(facecolor='white', edgecolor='#415062', linewidth=0.9),
            whiskerprops=dict(color='#415062', linewidth=0.9),
            medianprops=dict(color=PLOT_COLORS['truth'], linewidth=1.1),
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(methods, rotation=8)
    _style_axes(ax, ylabel='风险敏感目标值', title='稳健性分布')
    _panel_label(ax, '(c)')

    # (d) 参数收敛区域
    ax = axes[1, 1]
    cmap = plt.cm.Set2(np.linspace(0, 1, max(1, len(methods))))
    for method, color in zip(methods, cmap):
        runs = method_run_data[method].get('runs', [])
        if not runs:
            continue
        kp_vals = np.asarray([r['Kp'] for r in runs], dtype=float)
        kd_vals = np.asarray([r['Kd'] for r in runs], dtype=float)
        ax.scatter(kp_vals, kd_vals, s=85, color=color, alpha=0.72, edgecolors='white', linewidth=0.6, label=method)
        ax.scatter(np.mean(kp_vals), np.mean(kd_vals), s=155, marker='X', color=color, edgecolors='#1f252d', linewidth=0.8)
    _style_axes(ax, xlabel='Kp', ylabel='Kd', title='参数收敛区域')
    _panel_label(ax, '(d)')
    ax.set_xlim(bounds[0][0], bounds[1][0])
    ax.set_ylim(bounds[0][1], bounds[1][1])
    ax.legend(loc='upper right', fontsize=8.8)

    return _finish_figure(fig, title)
