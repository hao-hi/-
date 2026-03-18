"""
可视化模块
提供各种绘图功能
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors, ticker
from matplotlib.patches import FancyBboxPatch
from core_utils import quat_angle_errors_deg

# 解决中文和负号显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'sans-serif']
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'STSong']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 120
matplotlib.rcParams['savefig.dpi'] = 260
matplotlib.rcParams['savefig.facecolor'] = 'white'
matplotlib.rcParams['figure.facecolor'] = 'white'
matplotlib.rcParams['axes.facecolor'] = '#fbfcfe'
matplotlib.rcParams['axes.titleweight'] = 'semibold'
matplotlib.rcParams['axes.labelsize'] = 11.5
matplotlib.rcParams['axes.linewidth'] = 0.9
matplotlib.rcParams['axes.axisbelow'] = True
matplotlib.rcParams['axes.labelcolor'] = '#243447'
matplotlib.rcParams['text.color'] = '#1f2a36'
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
matplotlib.rcParams['axes.formatter.use_mathtext'] = False


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
    'minor_grid': '#edf1f6',
    'panel_bg': '#fbfcfe',
    'panel_edge': '#d6dee8',
    'text_muted': '#5f6c79',
}
_TRAPEZOID = getattr(np, "trapezoid", np.trapz)


def _curve_integral(y, x):
    """统一的梯形积分兼容层。"""
    return float(_TRAPEZOID(y, x))


def _style_axes(ax, xlabel=None, ylabel=None, title=None):
    """统一坐标轴样式。"""
    ax.set_facecolor(PLOT_COLORS['panel_bg'])
    ax.grid(True, color=PLOT_COLORS['grid'], linewidth=0.84, alpha=0.82)
    ax.minorticks_on()
    ax.grid(which='minor', color=PLOT_COLORS['minor_grid'], linewidth=0.48, alpha=0.92)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#94a0ad')
    ax.spines['bottom'].set_color('#94a0ad')
    ax.spines['left'].set_linewidth(0.92)
    ax.spines['bottom'].set_linewidth(0.92)
    ax.tick_params(colors='#33404d', which='major', width=0.82, length=4.1)
    ax.tick_params(colors='#6b7682', which='minor', width=0.55, length=2.2)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title, fontsize=13.2, fontweight='semibold', pad=11, color='#1f2a36')


def _finish_figure(fig, title=None):
    """统一图像收尾。"""
    fig.patch.set_facecolor('white')
    skip_tight_layout = bool(getattr(fig, "_skip_tight_layout", False))
    if title:
        fig.suptitle(title, fontsize=16.1, fontweight='semibold', y=0.992, color='#17212b')
        if skip_tight_layout:
            fig.subplots_adjust(top=0.915, left=0.06, right=0.975, bottom=0.07)
        else:
            fig.tight_layout(rect=(0, 0, 1, 0.973))
    else:
        if skip_tight_layout:
            fig.subplots_adjust(left=0.06, right=0.975, bottom=0.07, top=0.96)
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
        fontsize=10.3,
        fontweight='semibold',
        color='#223243',
        bbox=dict(boxstyle='round,pad=0.20', facecolor='white', edgecolor=PLOT_COLORS['panel_edge'], alpha=0.98),
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
        fontsize=8.75,
        color='#243447',
        linespacing=1.35,
        bbox=dict(boxstyle='round,pad=0.34', facecolor='white', edgecolor=PLOT_COLORS['panel_edge'], alpha=0.96),
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


def _optimizer_method_colors(method_names):
    """为优化器名称分配稳定、论文风格的配色。"""
    base_colors = {
        'GridSearch': '#2f6ea5',
        'NelderMead': '#e67e22',
        'SimAnneal': '#2f9d44',
        'PSO': '#d64541',
        'RandomSearch': '#8e63b6',
    }
    fallback = plt.cm.tab10(np.linspace(0.0, 1.0, max(1, len(method_names))))
    color_map = {}
    for idx, method in enumerate(method_names):
        color_map[method] = base_colors.get(method, mcolors.to_hex(fallback[idx]))
    return color_map


def _annotate_series_end(ax, x, y, text, color):
    """在曲线终点添加简洁标注。"""
    ax.scatter([x], [y], s=28, color=color, zorder=6)
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(6, 6),
        textcoords='offset points',
        fontsize=7.9,
        color=color,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, alpha=0.88),
    )


def _style_legend(ax, loc='best', ncol=1, fontsize=8.9, anchor=None):
    """统一图例边框和排版。"""
    legend = ax.legend(loc=loc, ncol=ncol, fontsize=fontsize, frameon=True, bbox_to_anchor=anchor)
    if legend is None:
        return None
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor(PLOT_COLORS['panel_edge'])
    frame.set_linewidth(0.9)
    frame.set_alpha(0.96)
    return legend


def _set_y_margin(ax, values, lower_pad=0.06, upper_pad=0.14, floor_zero=False):
    """为曲线图留出顶部注释空间，避免终点标注贴边。"""
    arrays = [np.asarray(val, dtype=float).reshape(-1) for val in values]
    finite_arrays = [arr[np.isfinite(arr)] for arr in arrays if arr.size]
    finite_arrays = [arr for arr in finite_arrays if arr.size]
    if not finite_arrays:
        return

    merged = np.concatenate(finite_arrays)
    y_min = float(np.min(merged))
    y_max = float(np.max(merged))
    span = max(y_max - y_min, 1e-9)
    low = y_min - lower_pad * span
    high = y_max + upper_pad * span
    if floor_zero:
        low = min(low, 0.0)
        low = max(0.0, low)
    ax.set_ylim(low, high)


def _draw_summary_cards(ax, entries, accent_color=None):
    """在仪表板顶部绘制简洁摘要卡片。"""
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    if not entries:
        return

    if accent_color is not None:
        ax.plot([0.012, 0.988], [0.94, 0.94], color=accent_color, linewidth=2.2, alpha=0.92, solid_capstyle='round')

    card_gap = 0.022
    card_width = (1.0 - card_gap * (len(entries) + 1)) / len(entries)
    x_cursor = card_gap
    for title, value, note in entries:
        patch = FancyBboxPatch(
            (x_cursor, 0.10),
            card_width,
            0.72,
            boxstyle='round,pad=0.012,rounding_size=0.018',
            linewidth=0.95,
            edgecolor=PLOT_COLORS['panel_edge'],
            facecolor='#fbfcfe',
            mutation_aspect=1.0,
        )
        ax.add_patch(patch)
        ax.text(x_cursor + 0.03, 0.66, str(title), fontsize=8.5, color=PLOT_COLORS['text_muted'], ha='left', va='center')
        ax.text(x_cursor + 0.03, 0.43, str(value), fontsize=13.0, fontweight='semibold', color='#1f2a36', ha='left', va='center')
        if note:
            ax.text(x_cursor + 0.03, 0.21, str(note), fontsize=8.1, color=PLOT_COLORS['text_muted'], ha='left', va='center')
        x_cursor += card_width + card_gap


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


def _tail_window_bounds(t_hist, tail_ratio=0.15, min_duration=1.0):
    """
    计算结果尾段的起止时间，便于突出稳态统计区间。
    """
    t_arr = np.asarray(t_hist, dtype=float).reshape(-1)
    if t_arr.size == 0:
        return None
    if t_arr.size == 1:
        return float(t_arr[0]), float(t_arr[0])

    total_duration = float(t_arr[-1] - t_arr[0])
    tail_duration = max(float(min_duration), total_duration * float(tail_ratio))
    start_time = max(float(t_arr[0]), float(t_arr[-1] - tail_duration))
    return start_time, float(t_arr[-1])


def _safe_relative_change_percent(baseline, candidate, zero_tol=1e-9):
    """
    计算 candidate 相对 baseline 的改变量百分比；负值表示 candidate 更小。
    """
    baseline = float(baseline)
    candidate = float(candidate)
    if abs(baseline) <= zero_tol:
        if abs(candidate - baseline) <= zero_tol:
            return 0.0
        return np.nan
    return (candidate - baseline) / baseline * 100.0


def _control_profile(u_hist, u_limit=None):
    """
    统一派生控制输入范数、单轴峰值和饱和掩码。
    """
    u_arr = np.asarray(u_hist, dtype=float)
    if u_arr.ndim != 2 or u_arr.shape[1] != 3:
        raise ValueError("u_hist 必须是 N×3 控制力矩数组")

    u_norm = np.linalg.norm(u_arr, axis=1)
    u_axis_peak = np.max(np.abs(u_arr), axis=1)

    if u_limit is None:
        axis_limit = float(np.max(u_axis_peak)) if u_axis_peak.size else 0.0
    else:
        axis_limit = float(u_limit)

    norm_limit = float(np.sqrt(u_arr.shape[1]) * axis_limit) if axis_limit > 0.0 else 0.0
    sat_mask = (
        u_axis_peak >= 0.98 * axis_limit
        if axis_limit > 0.0 else np.zeros(u_axis_peak.shape, dtype=bool)
    )
    return {
        'u_norm': u_norm,
        'u_axis_peak': u_axis_peak,
        'axis_limit': axis_limit,
        'norm_limit': norm_limit,
        'sat_mask': sat_mask,
        'peak_norm': float(np.max(u_norm)) if u_norm.size else 0.0,
        'peak_axis': float(np.max(u_axis_peak)) if u_axis_peak.size else 0.0,
    }


def _apply_score_axis_scaling(ax, arrays, threshold=20.0):
    """
    当目标值跨越范围过大时自动改用对数轴，提升可读性。
    """
    positive_values = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=float)
        finite = arr[np.isfinite(arr) & (arr > 0.0)]
        if finite.size:
            positive_values.append(finite)

    if not positive_values:
        return False

    merged = np.concatenate(positive_values)
    if merged.size < 2:
        return False

    dynamic_ratio = float(np.max(merged) / max(np.min(merged), 1e-12))
    if dynamic_ratio >= float(threshold):
        ax.set_yscale('log')
        return True
    return False


def _build_score_norm(values, upper_quantile=92.0):
    """
    为优化目标景观构造更稳健的颜色映射，减弱极端离群点影响。
    """
    score_arr = np.asarray(values, dtype=float)
    finite = score_arr[np.isfinite(score_arr)]
    if finite.size == 0:
        return None

    vmin = float(np.min(finite))
    vmax = float(np.percentile(finite, upper_quantile))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.max(finite))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    if vmin > 0.0 and vmax / max(vmin, 1e-12) >= 20.0:
        return mcolors.LogNorm(vmin=max(vmin, 1e-6), vmax=vmax)
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


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
    control_profile = _control_profile(u_hist, umax)
    
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
    u_norm = control_profile['u_norm']
    axs[3].plot(t_hist, u_norm, color=PLOT_COLORS['truth'], linewidth=2.2, label='||u||')
    if umax is not None:
        axs[3].axhline(
            y=control_profile['norm_limit'],
            color=PLOT_COLORS['limit'],
            linestyle='--',
            alpha=0.8,
            label=f'范数上界 {control_profile["norm_limit"]:.3f}',
        )
        if np.any(control_profile['sat_mask']):
            axs[3].fill_between(
                t_hist,
                0.0,
                u_norm,
                where=control_profile['sat_mask'],
                color=PLOT_COLORS['warning'],
                alpha=0.14,
            )
    _style_axes(axs[3], xlabel='时间 (s)', ylabel='||u|| (N·m)')
    _apply_time_axis(axs[3], t_hist)
    _panel_label(axs[3], '(d)')
    _annotate_series_end(axs[3], t_hist[-1], u_norm[-1], f"{u_norm[-1]:.3f}", PLOT_COLORS['truth'])
    _metric_box(
        axs[3],
        [
            f'积分能耗 = {_curve_integral(u_norm, t_hist):.3f} N·m·s',
            f'单轴峰值 = {control_profile["peak_axis"]:.3f} N·m',
        ],
        loc='upper right',
    )
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
    control_profile = _control_profile(u_hist, umax)
    
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
    u_norm = control_profile['u_norm']
    ax3.plot(t_hist, u_norm, color=PLOT_COLORS['truth'], linewidth=2.2, label='||u||')
    if umax is not None:
        ax3.axhline(
            y=control_profile['norm_limit'],
            color=PLOT_COLORS['limit'],
            linestyle='--',
            alpha=0.8,
            label=f'范数上界 {control_profile["norm_limit"]:.3f}',
        )
        if np.any(control_profile['sat_mask']):
            ax3.fill_between(
                t_hist,
                0.0,
                u_norm,
                where=control_profile['sat_mask'],
                color=PLOT_COLORS['warning'],
                alpha=0.14,
                label='轴向饱和区',
            )
    _style_axes(ax3, xlabel='时间 (s)', ylabel='||u|| (N·m)')
    _apply_time_axis(ax3, t_hist)
    _panel_label(ax3, '(c)')
    _metric_box(
        ax3,
        [
            f'能耗 = {_curve_integral(u_norm, t_hist):.3f} N·m·s',
            f'单轴峰值 = {control_profile["peak_axis"]:.3f} N·m',
        ],
        loc='upper right',
    )
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
    t = np.asarray(results_pd['t'], dtype=float)
    pd_settle = float(results_pd.get('settle_time', t[-1]))
    adrc_settle = float(results_adrc.get('settle_time', t[-1]))
    pd_control = _control_profile(results_pd['u'], umax)
    adrc_control = _control_profile(results_adrc['u'], umax)
    fig = plt.figure(figsize=(14.9, 11.0))
    fig._skip_tight_layout = True
    gs = fig.add_gridspec(3, 2, height_ratios=[0.28, 1.0, 1.0], hspace=0.25, wspace=0.16)
    ax_summary = fig.add_subplot(gs[0, :])
    axes = np.array([
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
        [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])],
    ], dtype=object)
    _draw_summary_cards(
        ax_summary,
        [
            ('误差收敛', f'PD {pd_settle:.2f} s', f'ADRC {adrc_settle:.2f} s'),
            ('末端误差', f'PD {results_pd.get("final_error", np.nan):.2f} deg', f'ADRC {results_adrc.get("final_error", np.nan):.2f} deg'),
            ('控制能耗', f'PD {results_pd.get("effort", np.nan):.3f}', f'ADRC {results_adrc.get("effort", np.nan):.3f}'),
            ('尾段 RMS', f'PD {results_pd.get("steady_state_rms", np.nan):.2f}', f'ADRC {results_adrc.get("steady_state_rms", np.nan):.2f}'),
        ],
        accent_color=PLOT_COLORS['adrc'],
    )

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
    tail_bounds = _tail_window_bounds(t)
    if tail_bounds is not None:
        tail_start, tail_end = tail_bounds
        ax.axvspan(tail_start, tail_end, color=PLOT_COLORS['soft_fill'], alpha=0.45, zorder=0)
    _style_axes(ax, xlabel='时间 (s)', ylabel='姿态误差 (度)', title='姿态误差对比')
    _apply_time_axis(ax, t)
    _panel_label(ax, '(a)')
    _set_y_margin(ax, [err_pd, err_adrc, [1.0]], floor_zero=True)
    _annotate_series_end(ax, t[-1], err_pd[-1], f"{err_pd[-1]:.2f}", PLOT_COLORS['pd'])
    _annotate_series_end(ax, t[-1], err_adrc[-1], f"{err_adrc[-1]:.2f}", PLOT_COLORS['adrc'])
    _metric_box(
        ax,
        [
            f'PD: Ts={pd_settle:.2f}s, Final={err_pd[-1]:.2f}deg',
            f'ADRC: Ts={adrc_settle:.2f}s, Final={err_adrc[-1]:.2f}deg',
            f'尾段RMS: PD={results_pd.get("steady_state_rms", np.nan):.2f}, ADRC={results_adrc.get("steady_state_rms", np.nan):.2f}',
        ],
        loc='upper right',
    )
    _style_legend(ax, loc='center right')

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
    _set_y_margin(ax, [w_pd_norm, w_adrc_norm], floor_zero=True)
    _metric_box(
        ax,
        [
            f'PD 峰值 = {np.max(w_pd_norm):.2f}',
            f'ADRC 峰值 = {np.max(w_adrc_norm):.2f}',
        ],
        loc='upper right',
    )
    _style_legend(ax, loc='lower right')

    ax = axes[1, 0]
    u_pd_norm = pd_control['u_norm']
    u_adrc_norm = adrc_control['u_norm']
    ax.fill_between(t, u_pd_norm, color=PLOT_COLORS['pd'], alpha=0.06)
    ax.fill_between(t, u_adrc_norm, color=PLOT_COLORS['adrc'], alpha=0.06)
    ax.plot(t, u_pd_norm, color=PLOT_COLORS['pd'], linewidth=2.2, label='PD')
    ax.plot(t, u_adrc_norm, color=PLOT_COLORS['adrc'], linewidth=2.2, linestyle='--', label='ADRC')
    ax.axhline(
        y=pd_control['norm_limit'],
        color=PLOT_COLORS['limit'],
        linestyle=':',
        alpha=0.9,
        label=f'范数上界 {pd_control["norm_limit"]:.3f}',
    )
    if umax is not None:
        if np.any(pd_control['sat_mask']):
            ax.fill_between(t, 0.0, u_pd_norm, where=pd_control['sat_mask'], color=PLOT_COLORS['pd'], alpha=0.12)
        if np.any(adrc_control['sat_mask']):
            ax.fill_between(t, 0.0, u_adrc_norm, where=adrc_control['sat_mask'], color=PLOT_COLORS['adrc'], alpha=0.12)
    _style_axes(ax, xlabel='时间 (s)', ylabel='||u|| (N·m)', title='控制力矩对比')
    _apply_time_axis(ax, t)
    _panel_label(ax, '(c)')
    _set_y_margin(ax, [u_pd_norm, u_adrc_norm, [pd_control['norm_limit']]], floor_zero=True)
    _metric_box(
        ax,
        [
            f'PD 能耗 = {results_pd["effort"]:.3f}, 单轴峰值 = {results_pd.get("peak_axis_torque", pd_control["peak_axis"]):.3f}',
            f'ADRC 能耗 = {results_adrc["effort"]:.3f}, 单轴峰值 = {results_adrc.get("peak_axis_torque", adrc_control["peak_axis"]):.3f}',
        ],
        loc='upper right',
    )
    _style_legend(ax, loc='center right')

    ax = axes[1, 1]
    metric_defs = [
        ('调节时间', 'settle_time'),
        ('稳态误差', 'final_error'),
        ('尾段RMS', 'steady_state_rms'),
        ('ITAE', 'ITAE'),
        ('峰值单轴力矩', 'peak_axis_torque'),
    ]
    labels = [item[0] for item in metric_defs]
    improvements = np.array([
        -_safe_relative_change_percent(results_pd[key], results_adrc[key])
        for _, key in metric_defs
    ], dtype=float)
    y_pos = np.arange(len(labels))
    colors = [
        PLOT_COLORS['success'] if np.isfinite(val) and val >= 0.0 else PLOT_COLORS['warning']
        for val in improvements
    ]
    ax.barh(y_pos, np.nan_to_num(improvements, nan=0.0), color=colors, alpha=0.82)
    ax.axvline(0.0, color=PLOT_COLORS['limit'], linewidth=1.0, linestyle='--')
    for idx, (_, key) in enumerate(metric_defs):
        pd_val = float(results_pd[key])
        adrc_val = float(results_adrc[key])
        impr = improvements[idx]
        impr_text = "N/A" if not np.isfinite(impr) else f"{impr:+.1f}%"
        text_x = 1.2 if not np.isfinite(impr) else impr + (1.8 if impr >= 0.0 else -1.8)
        ha = 'left' if (not np.isfinite(impr) or impr >= 0.0) else 'right'
        ax.text(
            text_x,
            idx,
            f'{impr_text} | {adrc_val:.2f}/{pd_val:.2f}',
            va='center',
            ha=ha,
            fontsize=8.3,
            color='#223243',
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.2)
    lim = np.nanmax(np.abs(improvements)) if np.any(np.isfinite(improvements)) else 10.0
    lim = max(10.0, min(200.0, lim * 1.28))
    ax.set_xlim(-lim, lim)
    _style_axes(ax, xlabel='ADRC 相对 PD 改进 (%)', title='关键指标改进幅度')
    _panel_label(ax, '(d)')
    ax.grid(True, axis='x', color=PLOT_COLORS['grid'], linewidth=0.8, alpha=0.8)
    ax.minorticks_off()
    _metric_box(
        ax,
        [
            '正值表示 ADRC 更优（指标更小）',
            f'能耗差值 = {results_adrc["effort"] - results_pd["effort"]:+.3f} N·m·s',
            f'单轴峰值差值 = {results_adrc.get("peak_axis_torque", np.nan) - results_pd.get("peak_axis_torque", np.nan):+.3f} N·m',
        ],
        loc='upper left',
    )

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
    control_profile = _control_profile(u, results.get('u_limit'))

    w_norm = np.rad2deg(np.linalg.norm(w, axis=1))
    u_norm = control_profile['u_norm']
    bias_true_norm = np.linalg.norm(bias_true, axis=1)
    bias_est_norm = np.linalg.norm(bias_est, axis=1)
    fig = plt.figure(figsize=(14.8, 10.8))
    fig._skip_tight_layout = True
    gs = fig.add_gridspec(3, 2, height_ratios=[0.28, 1.0, 1.0], hspace=0.24, wspace=0.15)
    ax_summary = fig.add_subplot(gs[0, :])
    axes = np.array([
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
        [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])],
    ], dtype=object)
    _draw_summary_cards(
        ax_summary,
        [
            ('控制器', str(results.get('controller_type', 'PD')).upper(), '闭环过程总览'),
            ('误差收敛', f'{err[-1]:.2f} deg', f'调节时间 {settle_time:.2f} s'),
            ('角速度峰值', f'{np.max(w_norm):.2f} deg/s', f'末端 {w_norm[-1]:.2f} deg/s'),
            ('控制输入', f'{_curve_integral(u_norm, t):.3f}', f'峰值单轴 {results.get("peak_axis_torque", control_profile["peak_axis"]):.3f} N·m'),
        ],
        accent_color=main_color,
    )

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
    _set_y_margin(ax, [err, [error_ref_deg]], floor_zero=True)
    _annotate_series_end(ax, t[-1], err[-1], f"{err[-1]:.2f}", main_color)
    _metric_box(ax, [f'调节时间 = {settle_time:.2f} s', f'末端误差 = {err[-1]:.2f} deg'], loc='upper right')
    _style_legend(ax, loc='center right')

    ax = axes[0, 1]
    ax.plot(t, w_norm, color=PLOT_COLORS['z'], linewidth=2.1, label='角速度范数')
    _style_axes(ax, ylabel='||ω|| (deg/s)', title='角速度衰减过程')
    _apply_time_axis(ax, t)
    _panel_label(ax, '(b)')
    _set_y_margin(ax, [w_norm], floor_zero=True)
    _annotate_series_end(ax, t[-1], w_norm[-1], f"{w_norm[-1]:.2f}", PLOT_COLORS['z'])
    _metric_box(ax, [f'峰值 = {np.max(w_norm):.2f} deg/s', f'末端 = {w_norm[-1]:.2f} deg/s'], loc='upper right')
    _style_legend(ax, loc='lower right')

    ax = axes[1, 0]
    ax.plot(t, u_norm, color=PLOT_COLORS['truth'], linewidth=2.2, label='控制力矩范数')
    ax.axhline(
        control_profile['norm_limit'],
        color=PLOT_COLORS['limit'],
        linestyle='--',
        linewidth=1.1,
        label=f'范数上界 {control_profile["norm_limit"]:.2f}',
    )
    if np.any(control_profile['sat_mask']):
        ax.fill_between(t, 0.0, u_norm, where=control_profile['sat_mask'], color=PLOT_COLORS['adrc'], alpha=0.16, label='轴向饱和区')
    _style_axes(ax, xlabel='时间 (s)', ylabel='||u|| (N·m)', title='控制输入与饱和过程')
    _apply_time_axis(ax, t)
    _panel_label(ax, '(c)')
    _set_y_margin(ax, [u_norm, [control_profile['norm_limit']]], floor_zero=True)
    _annotate_series_end(ax, t[-1], u_norm[-1], f"{u_norm[-1]:.3f}", PLOT_COLORS['truth'])
    _metric_box(
        ax,
        [
            f'控制能耗 = {_curve_integral(u_norm, t):.3f} N·m·s',
            f'单轴峰值 = {results.get("peak_axis_torque", control_profile["peak_axis"]):.3f} N·m',
        ],
        loc='upper right',
    )
    _style_legend(ax, loc='center right')

    ax = axes[1, 1]
    ax.plot(t, bias_true_norm, color=PLOT_COLORS['truth'], linewidth=2.0, label='真实偏置范数')
    ax.plot(t, bias_est_norm, color=PLOT_COLORS['estimate'], linestyle='--', linewidth=2.0, label='估计偏置范数')
    _style_axes(ax, xlabel='时间 (s)', ylabel='||b|| (rad/s)', title='陀螺偏置估计过程')
    _apply_time_axis(ax, t)
    _panel_label(ax, '(d)')
    _set_y_margin(ax, [bias_true_norm, bias_est_norm], floor_zero=True)
    _annotate_series_end(ax, t[-1], bias_est_norm[-1], f"{bias_est_norm[-1]:.4f}", PLOT_COLORS['estimate'])
    _metric_box(ax, [f'偏置估计误差 = {abs(bias_true_norm[-1] - bias_est_norm[-1]):.4f}'], loc='upper right')
    _style_legend(ax, loc='lower right')

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
    axis_regressor_norms = np.asarray(
        results.get('axis_regressor_norms', np.full((len(t), 3), np.nan)),
        dtype=float,
    )
    regressor_min_sv_ref = results.get('regressor_min_sv_reference')
    scheme = str(results.get('inertia_estimator_scheme', 'Unknown'))
    labels = [('Jxx', PLOT_COLORS['x']), ('Jyy', PLOT_COLORS['y']), ('Jzz', PLOT_COLORS['z'])]
    update_ratio = 100.0 * float(np.mean(update_mask > 0.5)) if update_mask.size == t.size else 0.0
    err_norm = np.linalg.norm(inertia_err, axis=1)
    torque_norm = np.linalg.norm(u_hist, axis=1)
    accel_norm = np.linalg.norm(wdot_est, axis=1)
    finite_mask = np.isfinite(regressor_min_sv)
    update_binary = (update_mask > 0.5).astype(float)
    dt_local = float(np.median(np.diff(t))) if t.size > 1 else 1.0
    smooth_window = max(5, int(round(0.75 / max(dt_local, 1e-9))))
    kernel = np.ones(smooth_window, dtype=float) / smooth_window
    activity_ratio = 100.0 * np.convolve(update_binary, kernel, mode='same')
    event_times = t[update_binary > 0.5]
    axis_strength_mean = (
        np.nanmean(axis_regressor_norms[np.all(np.isfinite(axis_regressor_norms), axis=1)], axis=0)
        if axis_regressor_norms.ndim == 2 and axis_regressor_norms.shape[1] == 3 and np.any(np.all(np.isfinite(axis_regressor_norms), axis=1))
        else np.full(3, np.nan)
    )

    fig = plt.figure(figsize=(15.2, 11.2))
    fig._skip_tight_layout = True
    gs = fig.add_gridspec(3, 3, height_ratios=[0.27, 1.08, 1.0], hspace=0.26, wspace=0.18)
    ax_header = fig.add_subplot(gs[0, :])
    ax_main = fig.add_subplot(gs[1, :])
    ax_err = fig.add_subplot(gs[2, 0])
    ax_exc = fig.add_subplot(gs[2, 1])
    ax_sum = fig.add_subplot(gs[2, 2])
    _draw_summary_cards(
        ax_header,
        [
            ('辨识方案', scheme, '动力学参数在线更新'),
            ('最终误差范数', f'{err_norm[-1]:.4e}', f'RMSE {np.sqrt(np.mean(err_norm ** 2)):.4e}'),
            ('更新占比', f'{update_ratio:.1f}%', f'活跃时段 {event_times[0]:.2f}-{event_times[-1]:.2f}s' if event_times.size > 0 else '未检测到持续更新'),
            ('平均激励', f'||u|| {np.mean(torque_norm):.3f}', f'||wdot|| {np.mean(accel_norm):.3f}'),
        ],
        accent_color=PLOT_COLORS['adrc'],
    )

    _highlight_active_spans(ax_main, t, update_mask, PLOT_COLORS['success'], alpha=0.06)
    for idx, (label, color) in enumerate(labels):
        ax_main.plot(t, inertia_est[:, idx], color=color, linewidth=2.45, label=f'{label} 估计')
        ax_main.axhline(inertia_ref[idx], color=color, linestyle=(0, (4, 3)), linewidth=1.15, alpha=0.60)
        _annotate_series_end(ax_main, t[-1], inertia_est[-1, idx], f"{label} {inertia_est[-1, idx]:.4f}", color)
    _style_axes(ax_main, ylabel='惯量对角元 (kg·m^2)', title=f'辨识方案 | {scheme}')
    _panel_label(ax_main, '(a)')
    _apply_time_axis(ax_main, t)
    _set_y_margin(ax_main, [inertia_est[:, 0], inertia_est[:, 1], inertia_est[:, 2], inertia_ref], upper_pad=0.18)
    _style_legend(ax_main, loc='upper right', ncol=3, fontsize=9.0)
    _metric_box(
        ax_main,
        [
            f'参考惯量 = [{inertia_ref[0]:.3f}, {inertia_ref[1]:.3f}, {inertia_ref[2]:.3f}]',
            f'最终误差范数 = {err_norm[-1]:.4e}',
            f'参数更新占比 = {update_ratio:.1f}%',
        ],
        loc='lower right',
    )

    _highlight_active_spans(ax_err, t, update_mask, PLOT_COLORS['success'], alpha=0.05)
    ax_err.fill_between(t, 0.0, err_norm, color=PLOT_COLORS['soft_fill'], alpha=0.65, zorder=0)
    ax_err.plot(t, err_norm, color=PLOT_COLORS['truth'], linewidth=2.45, label='误差范数')
    for idx, (label, color) in enumerate(labels):
        ax_err.plot(t, inertia_err[:, idx], color=color, linewidth=1.55, alpha=0.95, label=f'{label}误差')
    ax_err.axhline(0.0, color='#9aa5b1', linestyle=':', linewidth=0.9)
    _style_axes(ax_err, xlabel='时间 (s)', ylabel='误差 (kg·m^2)', title='误差收敛')
    _panel_label(ax_err, '(b)')
    _apply_time_axis(ax_err, t)
    _set_y_margin(ax_err, [err_norm, inertia_err[:, 0], inertia_err[:, 1], inertia_err[:, 2]])
    _metric_box(
        ax_err,
        [
            f'RMSE = {np.sqrt(np.mean(err_norm ** 2)):.4e}',
            f'末值 = {err_norm[-1]:.4e}',
        ],
        loc='upper right',
    )
    _style_legend(ax_err, loc='lower right', fontsize=8.5, ncol=2)

    _highlight_active_spans(ax_exc, t, update_mask, PLOT_COLORS['success'], alpha=0.05)
    torque_scale = max(np.max(torque_norm), 1e-12)
    accel_scale = max(np.max(accel_norm), 1e-12)
    torque_plot = torque_norm / torque_scale
    accel_plot = accel_norm / accel_scale
    ax_exc.fill_between(t, 0.0, torque_plot, color=PLOT_COLORS['pd'], alpha=0.08)
    ax_exc.fill_between(t, 0.0, accel_plot, color=PLOT_COLORS['adrc'], alpha=0.08)
    ax_exc.plot(t, torque_plot, color=PLOT_COLORS['pd'], linewidth=2.0, label='控制力矩（归一化）')
    ax_exc.plot(t, accel_plot, color=PLOT_COLORS['adrc'], linewidth=2.0, label='角加速度（归一化）')
    ax_exc.plot(t, activity_ratio / 100.0, color=PLOT_COLORS['success'], linewidth=1.9, linestyle='--', label='局部更新率')
    if np.any(finite_mask):
        ax_pe = ax_exc.twinx()
        ax_pe.plot(
            t[finite_mask],
            regressor_min_sv[finite_mask],
            color=PLOT_COLORS['accent'],
            linewidth=1.7,
            alpha=0.9,
            label=r'$\sigma_{\min}(\Phi)$',
        )
        if regressor_min_sv_ref is not None and np.isfinite(regressor_min_sv_ref):
            ax_pe.axhline(float(regressor_min_sv_ref), color=PLOT_COLORS['accent'], linestyle=':', linewidth=1.0, alpha=0.8)
        ax_pe.set_ylabel(r'$\sigma_{\min}(\Phi)$', color=PLOT_COLORS['accent'])
        ax_pe.tick_params(axis='y', colors=PLOT_COLORS['accent'])
        ax_pe.spines['top'].set_visible(False)
        ax_pe.spines['right'].set_color('#c98a38')
        ax_pe.grid(False)
    _style_axes(ax_exc, xlabel='时间 (s)', ylabel='归一化量级', title='激励与更新活动')
    _panel_label(ax_exc, '(c)')
    _apply_time_axis(ax_exc, t)
    _set_y_margin(ax_exc, [torque_plot, accel_plot, activity_ratio / 100.0], floor_zero=True)
    _metric_box(
        ax_exc,
        [
            f'平均 ||u|| = {np.mean(torque_norm):.3f}',
            f'平均 ||wdot|| = {np.mean(accel_norm):.3f}',
        ],
        loc='upper right',
    )
    _style_legend(ax_exc, loc='lower right', fontsize=8.3)

    final_abs_err = np.abs(inertia_err[-1])
    y_pos = np.arange(3)
    bar_colors = [color for _, color in labels]
    ax_sum.barh(y_pos, final_abs_err, color=bar_colors, alpha=0.82, height=0.5)
    ax_sum.set_yticks(y_pos)
    ax_sum.set_yticklabels([label for label, _ in labels], fontsize=10)
    ax_sum.invert_yaxis()
    for idx, val in enumerate(final_abs_err):
        ax_sum.text(val + max(np.max(final_abs_err) * 0.03, 1e-4), idx, f'{val:.4f}', va='center', ha='left', fontsize=8.8, color='#243447')
    _style_axes(ax_sum, xlabel='最终绝对误差 (kg·m^2)', title='辨识质量摘要')
    _panel_label(ax_sum, '(d)')
    ax_sum.minorticks_off()
    ax_sum.grid(True, axis='x', color=PLOT_COLORS['grid'], linewidth=0.8, alpha=0.8)
    x_hi = max(np.max(final_abs_err) * 1.35, 1e-3)
    ax_sum.set_xlim(0.0, x_hi)

    summary_lines = [
        f'最终误差范数 = {err_norm[-1]:.4e}',
        f'更新占比 = {update_ratio:.1f}%',
    ]
    if np.any(finite_mask):
        summary_lines.append(f'PE均值 = {np.nanmean(regressor_min_sv):.4e}')
    if np.all(np.isfinite(axis_strength_mean)):
        summary_lines.append(
            f'轴向激励 = [{axis_strength_mean[0]:.3f}, {axis_strength_mean[1]:.3f}, {axis_strength_mean[2]:.3f}]'
        )
    if event_times.size > 0:
        summary_lines.append(f'活跃区间 = {event_times[0]:.2f}s - {event_times[-1]:.2f}s')
    _metric_box(ax_sum, summary_lines, loc='lower right')

    return _finish_figure(fig, title)


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
            plt.colorbar(filled, ax=ax, label='风险敏感目标函数值（越小越好）')
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
            plt.colorbar(scatter, ax=ax, label='风险敏感目标函数值（越小越好）')
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
        plt.colorbar(scatter, ax=ax, label='风险敏感目标函数值（越小越好）')

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
    _style_legend(axes[0], loc='upper right')
    
    _style_axes(axes[1], ylabel='||ω|| (deg/s)')
    _panel_label(axes[1], '(b)')
    _style_legend(axes[1], loc='upper right')
    
    _style_axes(axes[2], xlabel='时间 (s)', ylabel='||u|| (N·m)')
    _panel_label(axes[2], '(c)')
    _style_legend(axes[2], loc='upper right')

    for ax in axes:
        _apply_time_axis(ax, t)
        if ax.lines:
            _set_y_margin(ax, [line.get_ydata() for line in ax.lines], floor_zero=True)

    return _finish_figure(fig, title)


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
    control_profile = _control_profile(u, results.get('u_limit'))

    q_err = quat_angle_errors_deg(q_true, q_est)
    w_norm = np.rad2deg(np.linalg.norm(w, axis=1))
    u_norm = control_profile['u_norm']
    bias_err = np.linalg.norm(bias_true - bias_est, axis=1)
    tail_bounds = _tail_window_bounds(t)
    settle_time = float(results.get('settle_time', t[-1]))

    fig = plt.figure(figsize=(15.4, 12.0))
    fig._skip_tight_layout = True
    gs = fig.add_gridspec(4, 2, height_ratios=[0.30, 1.0, 1.0, 1.0], hspace=0.24, wspace=0.15)
    ax_summary = fig.add_subplot(gs[0, :])
    axes = np.array([
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
        [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])],
        [fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])],
    ], dtype=object)
    _draw_summary_cards(
        ax_summary,
        [
            ('控制器', str(results.get('controller_type', 'PD')).upper(), '单次闭环响应'),
            ('调节时间', f'{settle_time:.2f} s', f'末端误差 {results.get("final_error", err[-1]):.2f} deg'),
            ('控制能耗', f'{results.get("effort", _curve_integral(u_norm, t)):.3f}', f'峰值单轴 {results.get("peak_axis_torque", control_profile["peak_axis"]):.3f} N·m'),
            ('估计精度', f'{np.sqrt(np.mean(q_err ** 2)):.2f} deg', f'Bias RMSE {results.get("bias_rmse", np.sqrt(np.mean(bias_err ** 2))):.4f}'),
        ],
        accent_color=main_color,
    )

    ax = axes[0, 0]
    ax.plot(t, err, color=main_color, linewidth=2.2)
    ax.fill_between(t, err, color=main_color, alpha=0.10)
    ax.axhline(1.0, color=PLOT_COLORS['accent'], linestyle='--', linewidth=1.0)
    if settle_time <= t[-1]:
        ax.axvline(settle_time, color=PLOT_COLORS['limit'], linestyle=':', linewidth=1.1)
    if tail_bounds is not None:
        ax.axvspan(tail_bounds[0], tail_bounds[1], color=PLOT_COLORS['soft_fill'], alpha=0.42, zorder=0)
    _style_axes(ax, ylabel='姿态误差 (deg)', title='闭环姿态误差')
    _panel_label(ax, '(a)')
    _apply_time_axis(ax, t)
    _set_y_margin(ax, [err, [1.0]], floor_zero=True)
    _metric_box(
        ax,
        [
            f'Ts = {results["settle_time"]:.2f} s',
            f'Final = {results["final_error"]:.2f} deg',
            f'Tail RMS = {results.get("steady_state_rms", np.nan):.2f} deg',
        ],
    )

    ax = axes[0, 1]
    ax.plot(t, q_err, color=PLOT_COLORS['estimate'], linewidth=2.0)
    ax.fill_between(t, q_err, color=PLOT_COLORS['estimate'], alpha=0.10)
    if tail_bounds is not None:
        ax.axvspan(tail_bounds[0], tail_bounds[1], color=PLOT_COLORS['soft_fill'], alpha=0.38, zorder=0)
    _style_axes(ax, ylabel='估计误差 (deg)', title='姿态估计误差')
    _panel_label(ax, '(b)')
    _apply_time_axis(ax, t)
    _set_y_margin(ax, [q_err], floor_zero=True)
    _metric_box(
        ax,
        [
            f'RMSE = {np.sqrt(np.mean(q_err ** 2)):.2f} deg',
            f'Final = {q_err[-1]:.2f} deg',
        ],
    )

    ax = axes[1, 0]
    ax.plot(t, w_norm, color=PLOT_COLORS['z'], linewidth=2.0)
    _style_axes(ax, ylabel='||ω|| (deg/s)', title='角速度衰减')
    _panel_label(ax, '(c)')
    _apply_time_axis(ax, t)
    _set_y_margin(ax, [w_norm], floor_zero=True)
    _metric_box(
        ax,
        [
            f'Max = {results.get("peak_rate_deg", np.max(w_norm)):.2f}',
            f'Final = {w_norm[-1]:.2f}',
        ],
    )

    ax = axes[1, 1]
    ax.plot(t, u_norm, color=PLOT_COLORS['truth'], linewidth=2.0)
    ax.axhline(control_profile['norm_limit'], color=PLOT_COLORS['limit'], linestyle='--', linewidth=1.0)
    if np.any(control_profile['sat_mask']):
        ax.fill_between(t, 0.0, u_norm, where=control_profile['sat_mask'], color=PLOT_COLORS['warning'], alpha=0.18)
    _style_axes(ax, ylabel='||u|| (N·m)', title='控制输入范数')
    _panel_label(ax, '(d)')
    _apply_time_axis(ax, t)
    _set_y_margin(ax, [u_norm, [control_profile['norm_limit']]], floor_zero=True)
    _metric_box(
        ax,
        [
            f'Effort = {results["effort"]:.3f}',
            f'Peak axis = {results.get("peak_axis_torque", control_profile["peak_axis"]):.3f} N·m',
            f'Usage = {results.get("torque_usage_ratio", 0.0) * 100:.1f}%',
        ],
    )

    ax = axes[2, 0]
    ax.plot(t, np.linalg.norm(bias_true, axis=1), color=PLOT_COLORS['truth'], linewidth=1.8, label='真实偏置')
    ax.plot(t, np.linalg.norm(bias_est, axis=1), color=PLOT_COLORS['estimate'], linestyle='--', linewidth=1.8, label='估计偏置')
    _style_axes(ax, xlabel='时间 (s)', ylabel='||b|| (rad/s)', title='陀螺偏置跟踪')
    _panel_label(ax, '(e)')
    _apply_time_axis(ax, t)
    _set_y_margin(ax, [np.linalg.norm(bias_true, axis=1), np.linalg.norm(bias_est, axis=1)], floor_zero=True)
    _style_legend(ax, loc='lower right')

    ax = axes[2, 1]
    ax.plot(t, bias_err, color=PLOT_COLORS['adrc'], linewidth=2.0)
    ax.fill_between(t, bias_err, color=PLOT_COLORS['adrc'], alpha=0.10)
    _style_axes(ax, xlabel='时间 (s)', ylabel='偏置误差范数', title='估计误差收敛')
    _panel_label(ax, '(f)')
    _apply_time_axis(ax, t)
    _set_y_margin(ax, [bias_err], floor_zero=True)
    _metric_box(
        ax,
        [
            f'Final = {bias_err[-1]:.4f}',
            f'Mean = {np.mean(bias_err):.4f}',
            f'RMSE = {results.get("bias_rmse", np.nan):.4f}',
        ],
    )

    return _finish_figure(fig, title)


def plot_optimizer_report_dashboard(cache_dict, method_run_data, comparison_results, bounds, title="PD优化结果综合仪表板"):
    """
    生成优化结果综合仪表板，汇总景观、分布、收敛与参数分布。
    """
    fig = plt.figure(figsize=(15.4, 11.8))
    fig._skip_tight_layout = True
    gs = fig.add_gridspec(3, 2, height_ratios=[0.28, 1.0, 1.0], hspace=0.24, wspace=0.16)
    ax_header = fig.add_subplot(gs[0, :])
    axes = np.array([
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
        [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])],
    ], dtype=object)
    best_row = comparison_results[0] if comparison_results else {}
    best_method = best_row.get('method', 'N/A')
    _draw_summary_cards(
        ax_header,
        [
            ('最优算法', best_method, f'score {best_row.get("score_best", best_row.get("score", np.nan)):.3f}' if comparison_results else '无结果'),
            ('最优增益', f'Kp={best_row.get("Kp", np.nan):.3f}', f'Kd={best_row.get("Kd", np.nan):.3f}' if comparison_results else ''),
            ('评估规模', f'{len(cache_dict) if cache_dict else 0} 点', f'{len(method_run_data)} 种优化器'),
            ('性能摘要', f'Ts={best_row.get("settle_time", np.nan):.2f} s', f'Effort={best_row.get("effort", np.nan):.3f}' if comparison_results else ''),
        ],
        accent_color=PLOT_COLORS['pd'],
    )

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
        score_norm = _build_score_norm(score_arr)
        scatter = ax.scatter(
            kp_arr,
            kd_arr,
            c=score_arr,
            cmap='cividis',
            norm=score_norm,
            s=42,
            alpha=0.86,
            edgecolors='white',
            linewidth=0.35,
        )
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Objective (robust scale)', fontsize=9)
        elite_threshold = float(np.percentile(score_arr, 18.0))
        elite_mask = score_arr <= elite_threshold
        if np.any(elite_mask):
            ax.scatter(
                kp_arr[elite_mask],
                kd_arr[elite_mask],
                s=88,
                facecolors='none',
                edgecolors='white',
                linewidth=0.9,
                alpha=0.75,
                label='较优评估点',
            )
        best_idx = int(np.argmin(score_arr))
        ax.scatter([kp_arr[best_idx]], [kd_arr[best_idx]], marker='*', s=320, color=PLOT_COLORS['adrc'], edgecolors='white', linewidth=1.0)
        _metric_box(
            ax,
            [
                f'已评估点 = {kp_arr.size}',
                f'最优 = ({kp_arr[best_idx]:.2f}, {kd_arr[best_idx]:.2f})',
                f'较优点阈值 = 前 {np.mean(elite_mask) * 100:.0f}%',
            ],
            loc='upper left',
        )
    _style_axes(ax, xlabel='Kp', ylabel='Kd', title='参数景观')
    _panel_label(ax, '(a)')
    ax.set_xlim(bounds[0][0], bounds[1][0])
    ax.set_ylim(bounds[0][1], bounds[1][1])

    # (b) 收敛
    ax = axes[0, 1]
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
    log_scaled = _apply_score_axis_scaling(
        ax,
        [row.get('history', []) for row in comparison_results],
    )
    if comparison_results:
        _metric_box(
            ax,
            [
                f'最佳算法 = {best_method}',
                f'最佳 score = {comparison_results[0]["score_best"]:.3f}',
                'Y轴 = 对数' if log_scaled else 'Y轴 = 线性',
            ],
            loc='upper right',
        )
    _style_legend(ax, loc='upper right', fontsize=8.8)

    # (c) 目标值分布
    ax = axes[1, 0]
    methods = list(method_run_data.keys())
    if methods:
        positions = np.arange(1, len(methods) + 1)
        data = [np.asarray([r['score'] for r in method_run_data[m].get('runs', [])], dtype=float) for m in methods]
        cmap = plt.cm.Set2(np.linspace(0, 1, len(methods)))
        max_samples = max((arr.size for arr in data), default=0)
        if max_samples <= 2:
            means = np.asarray([np.nanmean(arr) if arr.size else np.nan for arr in data], dtype=float)
            stds = np.asarray([np.nanstd(arr) if arr.size > 1 else 0.0 for arr in data], dtype=float)
            ax.bar(positions, means, color=cmap, alpha=0.72, width=0.58, edgecolor='white', linewidth=0.8)
            ax.errorbar(positions, means, yerr=stds, fmt='none', ecolor='#33404d', elinewidth=1.0, capsize=4)
            for pos, arr, color in zip(positions, data, cmap):
                if arr.size:
                    jitter = np.linspace(-0.08, 0.08, arr.size)
                    ax.scatter(np.full(arr.size, pos) + jitter, arr, color=color, edgecolors='white', linewidth=0.6, s=46, zorder=4)
        else:
            violin = ax.violinplot(data, positions=positions, widths=0.75, showmeans=False, showextrema=False)
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
    if methods:
        mean_scores = [float(np.nanmean(arr)) if arr.size else np.nan for arr in data]
        _metric_box(
            ax,
            [
                f'最小均值 = {np.nanmin(mean_scores):.3f}',
                f'最大均值 = {np.nanmax(mean_scores):.3f}',
                '样本较少时回退为柱状+散点' if max_samples <= 2 else '分布 = 小提琴 + 箱线',
            ],
            loc='upper right',
        )

    # (d) 参数收敛区域
    ax = axes[1, 1]
    cmap = plt.cm.Set2(np.linspace(0, 1, max(1, len(methods))))
    all_kp = []
    all_kd = []
    for method, color in zip(methods, cmap):
        runs = method_run_data[method].get('runs', [])
        if not runs:
            continue
        kp_vals = np.asarray([r['Kp'] for r in runs], dtype=float)
        kd_vals = np.asarray([r['Kd'] for r in runs], dtype=float)
        all_kp.append(kp_vals)
        all_kd.append(kd_vals)
        ax.scatter(kp_vals, kd_vals, s=85, color=color, alpha=0.72, edgecolors='white', linewidth=0.6, label=method)
        ax.scatter(np.mean(kp_vals), np.mean(kd_vals), s=155, marker='X', color=color, edgecolors='#1f252d', linewidth=0.8)
    _style_axes(ax, xlabel='Kp', ylabel='Kd', title='参数收敛区域')
    _panel_label(ax, '(d)')
    if all_kp and all_kd:
        merged_kp = np.concatenate(all_kp)
        merged_kd = np.concatenate(all_kd)
        full_x_span = bounds[1][0] - bounds[0][0]
        full_y_span = bounds[1][1] - bounds[0][1]
        kp_span = max(np.ptp(merged_kp), full_x_span * 0.05, 1e-6)
        kd_span = max(np.ptp(merged_kd), full_y_span * 0.05, 1e-6)
        x_margin = kp_span * 0.35
        y_margin = kd_span * 0.35
        x0 = max(bounds[0][0], float(np.min(merged_kp) - x_margin))
        x1 = min(bounds[1][0], float(np.max(merged_kp) + x_margin))
        y0 = max(bounds[0][1], float(np.min(merged_kd) - y_margin))
        y1 = min(bounds[1][1], float(np.max(merged_kd) + y_margin))
        if (x1 - x0) / max(full_x_span, 1e-9) < 0.78:
            ax.set_xlim(x0, x1)
        else:
            ax.set_xlim(bounds[0][0], bounds[1][0])
        if (y1 - y0) / max(full_y_span, 1e-9) < 0.78:
            ax.set_ylim(y0, y1)
        else:
            ax.set_ylim(bounds[0][1], bounds[1][1])
        _metric_box(
            ax,
            [
                f'搜索边界 = [{bounds[0][0]:.1f}, {bounds[1][0]:.1f}] × [{bounds[0][1]:.1f}, {bounds[1][1]:.1f}]',
                f'聚集中心 = ({np.mean(merged_kp):.2f}, {np.mean(merged_kd):.2f})',
            ],
            loc='lower right',
        )
    else:
        ax.set_xlim(bounds[0][0], bounds[1][0])
        ax.set_ylim(bounds[0][1], bounds[1][1])
    _style_legend(ax, loc='upper right', fontsize=8.7)

    return _finish_figure(fig, title)


def plot_optimizer_convergence_statistics(method_run_data, comparison_results, title="优化器收敛统计曲线（均值±标准差）"):
    """
    绘制多优化器 best-so-far 收敛曲线的均值±标准差图。
    """
    if not comparison_results:
        return None

    ordered_methods = [row['method'] for row in comparison_results if row['method'] in method_run_data]
    if not ordered_methods:
        return None

    colors = _optimizer_method_colors(ordered_methods)
    fig, ax = plt.subplots(figsize=(12.6, 7.4))
    arrays_for_scaling = []
    run_count = 0

    for row in comparison_results:
        method = row['method']
        if method not in method_run_data:
            continue
        record = method_run_data[method]
        mean_hist = np.asarray(record.get('mean_hist', []), dtype=float)
        std_hist = np.asarray(record.get('std_hist', []), dtype=float)
        if mean_hist.size == 0:
            continue

        color = colors[method]
        iterations = np.arange(mean_hist.size, dtype=int)
        run_count = max(run_count, len(record.get('runs', [])))
        lower = np.maximum(mean_hist - std_hist, 1e-12)
        upper = mean_hist + std_hist

        ax.plot(
            iterations,
            mean_hist,
            color=color,
            linewidth=2.5,
            label=method,
            zorder=3,
        )
        ax.fill_between(
            iterations,
            lower,
            upper,
            color=color,
            alpha=0.14,
            linewidth=0.0,
            zorder=2,
        )
        ax.scatter(
            [iterations[-1]],
            [mean_hist[-1]],
            s=34,
            color=color,
            edgecolors='white',
            linewidth=0.8,
            zorder=4,
        )
        arrays_for_scaling.extend([mean_hist, lower, upper])

    _style_axes(ax, xlabel='Iteration', ylabel='Best-so-far objective', title=title)
    log_scaled = _apply_score_axis_scaling(ax, arrays_for_scaling)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(8, integer=True))
    ax.legend(loc='upper right', fontsize=9.4, ncol=1)

    best_row = comparison_results[0]
    _metric_box(
        ax,
        [
            f'统计样本数 n = {run_count}',
            f'最优算法 = {best_row["method"]}',
            f'最佳均值 score = {best_row["score"]:.3f}',
            'Y 轴 = 对数尺度' if log_scaled else 'Y 轴 = 线性尺度',
        ],
        loc='upper left',
    )
    return _finish_figure(fig, title)


def plot_optimizer_best_response_dashboard(method_run_data, comparison_results, title="不同优化器最优增益下的闭环响应对比"):
    """
    展示各优化器最优参数对应的姿态误差、角速度、控制输入与调节时间对比。
    """
    if not comparison_results:
        return None

    ordered_methods = [row['method'] for row in comparison_results if row['method'] in method_run_data]
    if not ordered_methods:
        return None

    colors = _optimizer_method_colors(ordered_methods)
    fig = plt.figure(figsize=(15.4, 11.0))
    fig._skip_tight_layout = True
    gs = fig.add_gridspec(3, 2, height_ratios=[0.28, 1.0, 1.0], hspace=0.24, wspace=0.16)
    ax_header = fig.add_subplot(gs[0, :])
    ax_err = fig.add_subplot(gs[1, 0])
    ax_rate = fig.add_subplot(gs[1, 1])
    ax_torque = fig.add_subplot(gs[2, 0])
    ax_settle = fig.add_subplot(gs[2, 1])

    time_ref = None
    settle_pairs = []
    score_pairs = []
    final_err_pairs = []

    for row in comparison_results:
        method = row['method']
        record = method_run_data.get(method)
        if not record:
            continue
        best_run = record.get('best_run', {})
        results = best_run.get('results')
        if not results:
            continue

        color = colors[method]
        t = np.asarray(results['t'], dtype=float)
        err = np.asarray(results['err'], dtype=float)
        w_norm = np.rad2deg(np.linalg.norm(np.asarray(results['w'], dtype=float), axis=1))
        control_profile = _control_profile(results['u'], results.get('u_limit'))
        u_norm = control_profile['u_norm']

        time_ref = t if time_ref is None else time_ref
        settle_pairs.append((method, float(row['settle_time'])))
        score_pairs.append((method, float(row['score'])))
        final_err_pairs.append((method, float(row['final_error'])))

        ax_err.plot(t, err, color=color, linewidth=2.2, label=method)
        ax_rate.plot(t, w_norm, color=color, linewidth=2.0, label=method)
        ax_torque.plot(t, u_norm, color=color, linewidth=2.0, label=method)

    best_method = comparison_results[0]['method']
    best_score = comparison_results[0]['score']
    best_final_err = comparison_results[0]['final_error']
    _draw_summary_cards(
        ax_header,
        [
            ('最优方法', best_method, f'score {best_score:.3f}'),
            ('最佳末端误差', f'{best_final_err:.3f} deg', f'Ts {comparison_results[0]["settle_time"]:.2f} s'),
            ('候选方法数', f'{len(ordered_methods)}', '闭环最优响应'),
            ('时间尺度', f'{time_ref[-1]:.2f} s' if time_ref is not None else 'N/A', '统一仿真时域'),
        ],
        accent_color=PLOT_COLORS['adrc'],
    )

    ax_err.axhline(1.0, color=PLOT_COLORS['accent'], linestyle='--', linewidth=1.0, alpha=0.9)
    _style_axes(ax_err, ylabel='姿态误差 (deg)', title='姿态误差收敛')
    _panel_label(ax_err, '(a)')
    _style_legend(ax_err, loc='upper right', fontsize=8.7, ncol=1)

    _style_axes(ax_rate, ylabel='||ω|| (deg/s)', title='角速度衰减')
    _panel_label(ax_rate, '(b)')
    _style_legend(ax_rate, loc='upper right', fontsize=8.7, ncol=1)

    _style_axes(ax_torque, xlabel='时间 (s)', ylabel='||u|| (N·m)', title='控制输入范数')
    _panel_label(ax_torque, '(c)')
    _style_legend(ax_torque, loc='upper right', fontsize=8.7, ncol=1)

    if time_ref is not None:
        _apply_time_axis(ax_err, time_ref)
        _apply_time_axis(ax_rate, time_ref)
        _apply_time_axis(ax_torque, time_ref)
    if ax_err.lines:
        _set_y_margin(ax_err, [line.get_ydata() for line in ax_err.lines], floor_zero=True)
    if ax_rate.lines:
        _set_y_margin(ax_rate, [line.get_ydata() for line in ax_rate.lines], floor_zero=True)
    if ax_torque.lines:
        _set_y_margin(ax_torque, [line.get_ydata() for line in ax_torque.lines], floor_zero=True)

    methods = [name for name, _ in settle_pairs]
    settle_values = np.asarray([value for _, value in settle_pairs], dtype=float)
    y_pos = np.arange(len(methods))
    bar_colors = [colors[name] for name in methods]
    ax_settle.barh(y_pos, settle_values, color=bar_colors, alpha=0.82, height=0.54)
    ax_settle.set_yticks(y_pos)
    ax_settle.set_yticklabels(methods)
    ax_settle.invert_yaxis()
    for idx, method in enumerate(methods):
        ax_settle.text(
            settle_values[idx] + max(np.max(settle_values) * 0.03, 1e-3),
            idx,
            f'{settle_values[idx]:.2f} s',
            va='center',
            ha='left',
            fontsize=8.8,
            color='#243447',
        )
    _style_axes(ax_settle, xlabel='调节时间 Ts (s)', title='最优参数性能摘要')
    _panel_label(ax_settle, '(d)')
    ax_settle.minorticks_off()
    ax_settle.grid(True, axis='x', color=PLOT_COLORS['grid'], linewidth=0.8, alpha=0.8)
    ax_settle.set_xlim(0.0, max(np.max(settle_values) * 1.25, 1.0))

    score_span = max([score for _, score in score_pairs]) - min([score for _, score in score_pairs]) if score_pairs else 0.0
    _metric_box(
        ax_settle,
        [
            f'最优方法 = {best_method}',
            f'最佳 score = {best_score:.3f}',
            f'最佳末端误差 = {best_final_err:.3f} deg',
            f'score 离散度 = {score_span:.3f}',
        ],
        loc='lower right',
    )
    return _finish_figure(fig, title)


def plot_optimizer_tradeoff_scatter(comparison_results, title="优化器性能权衡图"):
    """
    以调节时间和控制能耗为主轴，展示不同优化器的精度-能耗权衡。
    """
    if not comparison_results:
        return None

    methods = [row['method'] for row in comparison_results]
    colors = _optimizer_method_colors(methods)
    settle = np.asarray([row['settle_time'] for row in comparison_results], dtype=float)
    effort = np.asarray([row['effort'] for row in comparison_results], dtype=float)
    final_err = np.asarray([row['final_error'] for row in comparison_results], dtype=float)
    score = np.asarray([row['score'] for row in comparison_results], dtype=float)
    err_scale = (final_err - np.min(final_err)) / max(np.ptp(final_err), 1e-9)
    marker_sizes = 240.0 + 680.0 * err_scale

    fig, ax = plt.subplots(figsize=(11.8, 7.6))
    for idx, row in enumerate(comparison_results):
        method = row['method']
        ax.scatter(
            settle[idx],
            effort[idx],
            s=marker_sizes[idx],
            color=colors[method],
            alpha=0.82,
            edgecolors='white',
            linewidth=1.0,
            zorder=4,
        )
        ax.annotate(
            f'{method}\nscore={score[idx]:.2f}',
            xy=(settle[idx], effort[idx]),
            xytext=(8, 8),
            textcoords='offset points',
            fontsize=8.8,
            color='#243447',
            bbox=dict(boxstyle='round,pad=0.24', facecolor='white', edgecolor='#c7d0da', alpha=0.94),
        )

    best = comparison_results[0]
    ax.scatter(
        [best['settle_time']],
        [best['effort']],
        s=marker_sizes[0] * 1.18,
        facecolors='none',
        edgecolors='#17212b',
        linewidth=1.5,
        zorder=5,
    )
    _style_axes(ax, xlabel='调节时间 Ts (s)', ylabel='控制能耗 (N·m·s)', title=title)
    _metric_box(
        ax,
        [
            '气泡越大: 末端误差越大',
            f'最优算法 = {best["method"]}',
            f'最佳点 = ({best["settle_time"]:.2f} s, {best["effort"]:.3f})',
        ],
        loc='upper right',
    )
    x_margin = max(np.ptp(settle) * 0.16, 0.35)
    y_margin = max(np.ptp(effort) * 0.18, 0.04)
    ax.set_xlim(np.min(settle) - x_margin, np.max(settle) + x_margin)
    ax.set_ylim(max(0.0, np.min(effort) - y_margin), np.max(effort) + y_margin)
    return _finish_figure(fig, title)


def plot_optimizer_metric_heatmap(comparison_results, title="优化器指标热力图（列内归一化）"):
    """
    将关键指标按列归一化为热力图，便于快速比较各优化器优劣。
    """
    if not comparison_results:
        return None

    metric_specs = [
        ('score', 'Score', '{:.2f}'),
        ('settle_time', 'Ts (s)', '{:.2f}'),
        ('final_error', 'Final (deg)', '{:.2f}'),
        ('overshoot', 'Overshoot', '{:.3f}'),
        ('effort', 'Effort', '{:.3f}'),
        ('sat_ratio', 'Sat. ratio', '{:.4f}'),
    ]
    methods = [row['method'] for row in comparison_results]
    raw = np.asarray(
        [[float(row[key]) for key, _, _ in metric_specs] for row in comparison_results],
        dtype=float,
    )
    mins = np.min(raw, axis=0)
    spans = np.ptp(raw, axis=0)
    safe_spans = np.where(spans > 1e-12, spans, 1.0)
    normed = (raw - mins) / safe_spans
    normed[:, spans <= 1e-12] = 0.0

    fig, ax = plt.subplots(figsize=(11.8, 5.8))
    image = ax.imshow(normed, cmap='RdYlGn_r', aspect='auto', vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(image, ax=ax, fraction=0.042, pad=0.03)
    cbar.set_label('列内归一化值（越绿越优）', fontsize=9.1)

    ax.set_xticks(np.arange(len(metric_specs)))
    ax.set_xticklabels([label for _, label, _ in metric_specs], rotation=12, ha='right')
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_title(title, fontsize=13.2, fontweight='semibold', pad=12, color='#1f2a36')
    ax.set_facecolor('#fcfdff')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', which='both', length=0, colors='#33404d')
    ax.set_xticks(np.arange(-0.5, len(metric_specs), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(methods), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.2)
    ax.minorticks_off()

    for row_idx in range(len(methods)):
        for col_idx, (_, _, fmt) in enumerate(metric_specs):
            value = raw[row_idx, col_idx]
            color = '#12202b' if normed[row_idx, col_idx] < 0.58 else 'white'
            ax.text(
                col_idx,
                row_idx,
                fmt.format(value),
                ha='center',
                va='center',
                fontsize=8.8,
                color=color,
                fontweight='semibold',
            )

    _metric_box(
        ax,
        [
            '按指标列分别归一化',
            '绿色更优，红色较弱',
            f'最佳算法 = {comparison_results[0]["method"]}',
        ],
        loc='lower right',
    )
    return _finish_figure(fig)
