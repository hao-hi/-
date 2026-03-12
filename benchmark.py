"""
项目基准脚本
用于量化不同仿真配置下的运行时间
"""

import time
import numpy as np

from main import simulate_attitude_control, compare_pd_gain_optimizers


def _time_call(fn, repeats=3):
    samples = []
    last_result = None
    for _ in range(max(1, int(repeats))):
        start = time.perf_counter()
        last_result = fn()
        samples.append(time.perf_counter() - start)
    arr = np.asarray(samples, dtype=float)
    return {
        'mean_s': float(np.mean(arr)),
        'std_s': float(np.std(arr)),
        'min_s': float(np.min(arr)),
        'max_s': float(np.max(arr)),
        'last_result': last_result,
    }


def run_benchmarks(repeats=3):
    cases = [
        (
            'PD_no_tracker',
            lambda: simulate_attitude_control(
                T=1.5,
                dt=0.03,
                use_star_tracker=False,
                show_plots=False,
                controller_type='PD',
                seed=0,
            ),
        ),
        (
            'ADRC_tracker',
            lambda: simulate_attitude_control(
                T=1.5,
                dt=0.03,
                use_star_tracker=True,
                show_plots=False,
                controller_type='ADRC',
                seed=0,
            ),
        ),
        (
            'ADRC_tracker_RLS',
            lambda: simulate_attitude_control(
                T=1.5,
                dt=0.03,
                use_star_tracker=True,
                show_plots=False,
                controller_type='ADRC',
                seed=0,
                inertia_estimator_cfg={'scheme': 'RLS', 'lambda_factor': 0.98},
            ),
        ),
        (
            'PD_optimizer_quick',
            lambda: compare_pd_gain_optimizers(
                seed=0,
                show_plots=False,
                save_plots=False,
                benchmark_runs=1,
                objective_eval_seeds=[0],
                quick=True,
            ),
        ),
    ]

    results = {}
    print("=" * 72)
    print("项目性能基准")
    print("=" * 72)
    for name, fn in cases:
        stats = _time_call(fn, repeats=repeats)
        results[name] = stats
        print(
            f"{name:<20} mean={stats['mean_s']:.4f}s  "
            f"std={stats['std_s']:.4f}s  min={stats['min_s']:.4f}s  max={stats['max_s']:.4f}s"
        )
    return results


if __name__ == '__main__':
    run_benchmarks(repeats=3)
