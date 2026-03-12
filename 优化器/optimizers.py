import numpy as np
from concurrent.futures import ThreadPoolExecutor as Pool
from itertools import product


def clip_bounds(x, lo, hi):
    return np.minimum(np.maximum(np.asarray(x, dtype=float), lo), hi)


def reflect_bounds(x, lo, hi):
    x = np.asarray(x, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    span = hi - lo

    reflected = np.array(x, copy=True, dtype=float)
    active = span > 0.0
    if np.any(active):
        period = 2.0 * span[active]
        shifted = reflected[..., active] - lo[active]
        wrapped = np.mod(shifted, period)
        reflected[..., active] = lo[active] + np.where(
            wrapped <= span[active],
            wrapped,
            period - wrapped
        )
    if np.any(~active):
        reflected[..., ~active] = lo[~active]
    return reflected


def _evaluate_points(fun, points, parallel=False, workers=4, pool=None):
    if parallel:
        if pool is not None:
            return list(pool.map(fun, points))
        with Pool(max_workers=workers) as local_pool:
            return list(local_pool.map(fun, points))
    return [fun(p) for p in points]


def _latin_hypercube(rng, n_samples, dim):
    u = (np.arange(n_samples, dtype=float)[:, None] + rng.rand(n_samples, dim)) / max(1, n_samples)
    lhs = np.zeros((n_samples, dim), dtype=float)
    for d in range(dim):
        perm = rng.permutation(n_samples)
        lhs[:, d] = u[perm, d]
    return lhs


def grid_search(fun, lo, hi, n_per_dim=8, parallel=False, workers=4):
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    grids = [np.linspace(lo[i], hi[i], int(n_per_dim)) for i in range(len(lo))]
    points = [np.asarray(p, dtype=float) for p in product(*grids)]
    vals = _evaluate_points(fun, points, parallel=parallel, workers=workers)
    hist = np.asarray(vals, dtype=float)
    best_idx = int(np.argmin(hist))
    return points[best_idx].copy(), float(hist[best_idx]), hist


def random_search(fun, lo, hi, iters=60, seed=0, parallel=False, workers=4):
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    dim = len(lo)
    n = max(1, int(iters))
    rng = np.random.RandomState(seed)

    lhs = _latin_hypercube(rng, n, dim)
    samples = lo + lhs * (hi - lo)
    vals = _evaluate_points(fun, list(samples), parallel=parallel, workers=workers)
    hist = np.asarray(vals, dtype=float)
    best_idx = int(np.argmin(hist))
    return samples[best_idx].copy(), float(hist[best_idx]), hist


def nelder_mead(
    fun,
    x0,
    step=0.2,
    iters=80,
    alpha=1.0,
    gamma=2.0,
    rho=0.5,
    sigma=0.5,
    lo=None,
    hi=None,
    x_tol=1e-6,
    f_tol=1e-8,
):
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    if lo is not None and hi is not None:
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        x0 = reflect_bounds(x0, lo, hi)

    if np.isscalar(step):
        step_vec = np.full(n, float(step), dtype=float)
    else:
        step_vec = np.asarray(step, dtype=float)
        if step_vec.size != n:
            step_vec = np.full(n, float(np.mean(step_vec)), dtype=float)

    simplex = np.zeros((n + 1, n), dtype=float)
    simplex[0] = x0
    for i in range(n):
        e = np.zeros(n, dtype=float)
        e[i] = 1.0
        simplex[i + 1] = x0 + step_vec * e
    if lo is not None and hi is not None:
        simplex = reflect_bounds(simplex, lo, hi)

    vals = np.asarray([fun(x) for x in simplex], dtype=float)
    hist = [float(np.min(vals))]

    for _ in range(max(1, int(iters))):
        order = np.argsort(vals)
        simplex = simplex[order]
        vals = vals[order]
        hist.append(float(vals[0]))

        diam = np.max(np.linalg.norm(simplex - simplex[0], axis=1))
        fspread = float(np.max(np.abs(vals - vals[0])))
        if diam < x_tol and fspread < f_tol:
            break

        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + alpha * (centroid - simplex[-1])
        if lo is not None and hi is not None:
            xr = reflect_bounds(xr, lo, hi)
        fr = float(fun(xr))

        if vals[0] <= fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
            continue

        if fr < vals[0]:
            xe = centroid + gamma * (xr - centroid)
            if lo is not None and hi is not None:
                xe = reflect_bounds(xe, lo, hi)
            fe = float(fun(xe))
            if fe < fr:
                simplex[-1], vals[-1] = xe, fe
            else:
                simplex[-1], vals[-1] = xr, fr
            continue

        xc = centroid + rho * (simplex[-1] - centroid)
        if lo is not None and hi is not None:
            xc = reflect_bounds(xc, lo, hi)
        fc = float(fun(xc))
        if fc < vals[-1]:
            simplex[-1], vals[-1] = xc, fc
            continue

        for i in range(1, len(simplex)):
            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
            if lo is not None and hi is not None:
                simplex[i] = reflect_bounds(simplex[i], lo, hi)
            vals[i] = float(fun(simplex[i]))
        vals[0] = float(fun(simplex[0]))

    best_idx = int(np.argmin(vals))
    return simplex[best_idx].copy(), float(vals[best_idx]), np.asarray(hist, dtype=float)


def simulated_annealing(fun, lo, hi, iters=120, T0=1.0, decay=0.98, seed=0):
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    rng = np.random.RandomState(seed)

    x = lo + rng.rand(len(lo)) * (hi - lo)
    fx = float(fun(x))
    x_best = x.copy()
    f_best = fx
    hist = [f_best]
    T = float(T0)

    for k in range(max(1, int(iters))):
        scale = 0.25 * (0.5 + 0.5 * np.sqrt(max(T, 1e-8) / max(T0, 1e-8)))
        xn = reflect_bounds(x + (hi - lo) * scale * rng.randn(len(lo)), lo, hi)
        fn = float(fun(xn))
        dE = fn - fx
        if dE <= 0.0 or rng.rand() < np.exp(-dE / max(T, 1e-12)):
            x, fx = xn, fn
            if fx < f_best:
                x_best, f_best = x.copy(), fx
        hist.append(f_best)
        T *= float(decay)
        if (k + 1) % 40 == 0 and T < 1e-3:
            T = max(T, 0.05 * T0)

    return x_best, float(f_best), np.asarray(hist, dtype=float)


def pso(fun, lo, hi, iters=40, swarm=16, w=0.7, c1=1.4, c2=1.4, seed=0, parallel=False, workers=4):
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    dim = len(lo)
    rng = np.random.RandomState(seed)
    swarm = max(4, int(swarm))
    vmax = 0.2 * (hi - lo)

    lhs = _latin_hypercube(rng, swarm, dim)
    x = lo + lhs * (hi - lo)
    v = 0.05 * (hi - lo) * rng.randn(swarm, dim)
    p = x.copy()

    if parallel:
        with Pool(max_workers=workers) as pool:
            fp = np.asarray(_evaluate_points(fun, list(x), parallel=True, workers=workers, pool=pool), dtype=float)
            g_idx = int(np.argmin(fp))
            g = p[g_idx].copy()
            fg = float(fp[g_idx])
            hist = [fg]
            for _ in range(max(1, int(iters))):
                r1 = rng.rand(swarm, dim)
                r2 = rng.rand(swarm, dim)
                v = w * v + c1 * r1 * (p - x) + c2 * r2 * (g - x)
                v = np.minimum(np.maximum(v, -vmax), vmax)
                x = reflect_bounds(x + v, lo, hi)

                f = np.asarray(_evaluate_points(fun, list(x), parallel=True, workers=workers, pool=pool), dtype=float)
                improved = f < fp
                p[improved] = x[improved]
                fp[improved] = f[improved]

                best_idx = int(np.argmin(fp))
                if fp[best_idx] < fg:
                    g = p[best_idx].copy()
                    fg = float(fp[best_idx])
                hist.append(fg)
        return g, fg, np.asarray(hist, dtype=float)

    fp = np.asarray(_evaluate_points(fun, list(x), parallel=False, workers=workers), dtype=float)
    g_idx = int(np.argmin(fp))
    g = p[g_idx].copy()
    fg = float(fp[g_idx])
    hist = [fg]
    for _ in range(max(1, int(iters))):
        r1 = rng.rand(swarm, dim)
        r2 = rng.rand(swarm, dim)
        v = w * v + c1 * r1 * (p - x) + c2 * r2 * (g - x)
        v = np.minimum(np.maximum(v, -vmax), vmax)
        x = reflect_bounds(x + v, lo, hi)

        f = np.asarray(_evaluate_points(fun, list(x), parallel=False, workers=workers), dtype=float)
        improved = f < fp
        p[improved] = x[improved]
        fp[improved] = f[improved]

        best_idx = int(np.argmin(fp))
        if fp[best_idx] < fg:
            g = p[best_idx].copy()
            fg = float(fp[best_idx])
        hist.append(fg)
    return g, fg, np.asarray(hist, dtype=float)
