#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convergence Geometry — Experimental Mathematics

This script operationalizes a **theory-driven experimental program** for iterative
root-finding algorithms (Newton-type and variants). It produces *quantitative* fields
of algorithmic properties over initial conditions, runs statistical analyses, and
generates a per-run PDF report and machine-readable summaries.

Motivation & Theory Anchors
**Newton–Kantorovich & α-theory (Smale)**: Provide *sufficient*, local convergence
conditions in terms of β, γ (hence α=β·γ). They do not chart the empirical geometry
(order-of-convergence and contractivity) *outside* guaranteed neighborhoods. We map it.
**Complex dynamics (Fatou/Julia for Newton maps)**: Qualitative basin structure is known.
We add *quantitative fields* (empirical order p, contraction r), and test **scaling laws**
near basin boundaries (finite-size scaling).
**Conditioning & sensitivity**: We connect **geometric covariates** (distance to boundary
δ, nearest-root distance d0, crowding) with **observed algorithmic rates**.
**Universality**: We ask whether different polynomial ensembles and algorithms show the
*same* scaling exponents and collapse onto master curves (after rescaling).

What this program measures per initial condition z0
**p(z0)**: Empirical order of convergence estimated from the tail of errors |z_k - z_*|.
**r_last(z0), r_fit(z0)**: Effective contraction factors (last ratio and log-error slope).
**δ(z0)**: Distance to the nearest *basin boundary* in the grid (safety margin).
**d0(z0)**: Distance from z0 to the nearest analytic root (geometry wrt roots).
**β(z0)=|f|/|f'|,  γ(z0)≈|f''|/(2|f'|)**: α-like proxies at the start.

What analyses are performed
ECDFs for iterations, p, r (distributional shape).
**Binned regressions** of p and r vs δ and d0 with **bootstrap 95% CIs**.
**Finite-size scaling**: log–log slope fits for (2−p) vs δ and r vs δ near the boundary.
**Effect sizes** (Cohen’s d, Cliff’s delta) comparing methods/precisions (if multiple configs).
**Collapse utility** (optional): rescales δ by a degree-dependent factor to test universality;
outputs an R²-like score for curve overlap quality.

Outputs per run
pixels_<tag>.csv`     per-pixel dataset (initial conditions → properties).
summary_<tag>.json`   run-level stats (means/medians, scaling slopes).
report_<tag>.pdf`     PDF with fields, ECDFs, CI plots, log–log panels, and summary text.
manifest.json`        index of all runs in an invocation.

Plotting policy
**matplotlib only**, one chart per figure, **no seaborn**, **no color specification**
(satisfies external plotting constraints).

References
Kantorovich, L.V. (1948). Functional analysis and applied mathematics.
Smale, S. (1981). The fundamental theorem of algebra and complexity theory.
Traub, J. F. (1964). Iterative Methods for the Solution of Equations.
Hubbard, Schleicher, Sutherland (1994). How to find all roots of complex polynomials by Newton’s method.
"""

import argparse, json, math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages



# Problem Ensembles (polynomial generators)

def roots_on_circle(n: int, jitter: float = 0.0, rng: np.random.Generator = None, radius: float = 1.0) -> np.ndarray:
    """
    Equally spaced roots on the circle, with optional random phase jitter (Gaussian).
    Parameters
    
    n : int
        Number of roots (polynomial degree).
    jitter : float
        Stddev for random angular perturbations (radians).
    rng : np.random.Generator
        Random generator for reproducibility.
    radius : float
        Circle radius.
    Returns

    np.ndarray[complex], shape (n,)
    """
    if rng is None: rng = np.random.default_rng(0)
    base = np.array([np.exp(2j*np.pi*k/n) for k in range(n)], dtype=np.complex128) * radius
    if jitter <= 0: return base
    phases = rng.normal(0, jitter, size=n)
    return np.exp(1j*phases) * base


def clustered_roots(n: int, clusters: int = 2, spread: float = 0.01, rng: np.random.Generator = None, radius: float = 1.0) -> np.ndarray:
    """
    Place roots around a few cluster centers on a circle.
    """
    if rng is None: rng = np.random.default_rng(0)
    centers = np.array([np.exp(2j*np.pi*k/clusters) for k in range(clusters)]) * radius
    per = max(1, n // clusters)
    roots = []
    for c in centers:
        pert = (rng.normal(0, spread, per) + 1j*rng.normal(0, spread, per))
        roots.extend(list(c + pert))
    while len(roots) < n:
        roots.append(centers[0] + (rng.normal(0, spread) + 1j*rng.normal(0, spread)))
    return np.array(roots[:n], dtype=np.complex128)


def multiple_roots(n: int, multiplicity: int = 2, rng: np.random.Generator = None, radius: float = 1.0) -> np.ndarray:
    """
    Construct a polynomial with repeated roots (non-simple). Order of convergence will drop.
    """
    if rng is None: rng = np.random.default_rng(0)
    k = max(1, n // multiplicity)
    base = roots_on_circle(k, jitter=0.0, rng=rng, radius=radius)
    roots = np.repeat(base, multiplicity)[:n]
    return roots.astype(np.complex128)


def kac_like_roots(n: int, rng: np.random.Generator = None, radius: float = 1.0) -> np.ndarray:
    """
    Synthetic Kac-like ensemble: random roots in a disk (not a true Kac by coefficients,
    which would require root-finding; adequate for qualitative universality testing).
    """
    if rng is None: rng = np.random.default_rng(0)
    angles = rng.uniform(0, 2*np.pi, n)
    r = radius * np.sqrt(rng.uniform(0, 1, n))
    return (r * np.exp(1j*angles)).astype(np.complex128)


def poly_from_roots(roots: np.ndarray):
    """
    Build a monic polynomial f with specified complex roots and return f, f', f'' evaluators.
    """
    coeffs = np.poly(roots)  # monic up to numerical errors
    d1 = np.polyder(coeffs)
    d2 = np.polyder(coeffs, 2)

    def f(z): return np.polyval(coeffs, z)
    def df(z): return np.polyval(d1, z)
    def d2f(z): return np.polyval(d2, z)

    return f, df, d2f



# 2) Algorithm Registry (Newton-type updates)

def step_newton(z, f, df, lam=1.0):
    """Classic Newton step (order 2 near simple roots)."""
    return z - lam * f(z) / df(z)

def step_halley(z, f, df, d2f, lam=1.0):
    """Halley’s method (order 3 near simple roots)."""
    num = 2.0 * f(z) * df(z)
    den = 2.0 * (df(z)**2) - f(z) * d2f(z)
    return z - lam * num / den

def step_chebyshev(z, f, df, d2f, lam=1.0):
    """Chebyshev’s method (cubic order near simple roots)."""
    frac = f(z) / df(z)
    corr = 1.0 + 0.5 * (f(z) * d2f(z) / (df(z)**2))
    return z - lam * frac * corr

ALG_STEPS = {
    "newton":     lambda z,f,df,d2f,lam: step_newton(z,f,df,lam),
    "halley":     lambda z,f,df,d2f,lam: step_halley(z,f,df,d2f,lam),
    "chebyshev":  lambda z,f,df,d2f,lam: step_chebyshev(z,f,df,d2f,lam),
}


# Configuration & Helpers

@dataclass
class Config:
    ensemble: str            # problem family (circle, jittered, clustered, multiple, kac)
    degree: int              # polynomial degree
    damping: float           # λ in damped iteration
    precision: str           # complex128 or complex64
    grid: int                # grid resolution per axis
    xlim: Tuple[float,float]
    ylim: Tuple[float,float]
    max_iters: int
    algorithm: str           # newton / halley / chebyshev
    outdir: Path

def complex_dtype(precision: str):
    return np.complex128 if precision.lower() in ("c128","complex128","float64","double") else np.complex64

def boundary_mask(root_idx: np.ndarray) -> np.ndarray:
    """Boolean map where each pixel has at least one 4-neighbor with a different basin."""
    m, n = root_idx.shape
    B = np.zeros_like(root_idx, dtype=bool)
    for di,dj in [(1,0),(-1,0),(0,1),(0,-1)]:
        shifted = np.full_like(root_idx, -9999)
        if di == 1: shifted[1:,:] = root_idx[:-1,:]
        if di == -1: shifted[:-1,:] = root_idx[1:,:]
        if dj == 1: shifted[:,1:] = root_idx[:,:-1]
        if dj == -1: shifted[:,:-1] = root_idx[:,1:]
        B |= ((root_idx >= 0) & (shifted >= 0) & (root_idx != shifted))
    return B

def distance_to_boundary(B: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Chamfer-distance transform (2-pass) to approximate Euclidean distance to the nearest boundary.
    Returns physical distance in domain units.
    """
    H, W = B.shape
    INF = 10**9
    D = np.full((H,W), INF, dtype=float)
    D[B] = 0.0
    # forward pass
    for i in range(H):
        for j in range(W):
            if i>0:   D[i,j] = min(D[i,j], D[i-1,j] + 1.0)
            if j>0:   D[i,j] = min(D[i,j], D[i,j-1] + 1.0)
            if i>0 and j>0: D[i,j] = min(D[i,j], D[i-1,j-1] + math.sqrt(2))
            if i>0 and j<W-1: D[i,j] = min(D[i,j], D[i-1,j+1] + math.sqrt(2))
    # backward pass
    for i in range(H-1, -1, -1):
        for j in range(W-1, -1, -1):
            if i<H-1: D[i,j] = min(D[i,j], D[i+1,j] + 1.0)
            if j<W-1: D[i,j] = min(D[i,j], D[i,j+1] + 1.0)
            if i<H-1 and j<W-1: D[i,j] = min(D[i,j], D[i+1,j+1] + math.sqrt(2))
            if i<H-1 and j>0:   D[i,j] = min(D[i,j], D[i+1,j-1] + math.sqrt(2))
    # convert pixel to coordinate distance
    return D * math.sqrt(dx*dx + dy*dy)

def estimate_order_and_contraction(errors: List[float]) -> Tuple[float,float,float]:
    """
    Given per-iteration errors |z_k - z_*|, estimate:
      - p_hat : empirical order via p ≈ log(e_{k+1}/e_k) / log(e_k/e_{k-1})
      - r_last: last-step contraction e_{k+1} / e_k
      - r_fit : exp(slope) from linear fit log e_k ~ a + b k over the 4-point tail
    """
    errs = [e for e in errors if e>0 and np.isfinite(e)]
    if len(errs) < 4: return np.nan, np.nan, np.nan
    tail = np.array(errs[-4:], dtype=float)
    e1,e2,e3,e4 = tail
    num = np.log(e4/e3) if (e3>0 and e4>0) else np.nan
    den = np.log(e3/e2) if (e2>0 and e3>0) else np.nan
    p_hat = num/den if (np.isfinite(num) and np.isfinite(den) and den != 0) else np.nan
    r_last = (e4/e3) if (e3>0) else np.nan
    k = np.arange(len(tail))
    loge = np.log(tail)
    if np.all(np.isfinite(loge)):
        A = np.vstack([np.ones_like(k), k]).T
        coef, *_ = np.linalg.lstsq(A, loge, rcond=None)
        r_fit = float(np.exp(coef[1]))
    else:
        r_fit = np.nan
    return float(p_hat), float(r_last), float(r_fit)

def ecdf(values: np.ndarray):
    v = values[np.isfinite(values)]
    if v.size == 0: return np.array([]), np.array([])
    v = np.sort(v); y = np.linspace(0,1,len(v), endpoint=False)
    return v, y

def bootstrap_ci(values: np.ndarray, func, B: int = 400, alpha: float = 0.05, rng=None) -> Tuple[float,float]:
    if rng is None: rng = np.random.default_rng(0)
    vals = values[np.isfinite(values)]
    if vals.size == 0: return np.nan, np.nan
    stats = []
    n = vals.size
    for _ in range(B):
        sample = vals[rng.integers(0, n, size=n)]
        stats.append(func(sample))
    lo = np.percentile(stats, 100*alpha/2); hi = np.percentile(stats, 100*(1-alpha/2))
    return float(lo), float(hi)

def loglog_slope(x: np.ndarray, y: np.ndarray, top_frac: float = 0.15) -> Tuple[float,float,float]:
    """
    Fit slope of log y vs log x in the smallest-x region (near boundary).
    Returns (slope, intercept, exp(intercept)).
    """
    m = np.isfinite(x) & np.isfinite(y) & (x>0) & (y>0)
    if m.sum() < 10: return np.nan, np.nan, np.nan
    xv, yv = x[m], y[m]
    k = max(5, int(len(xv)*top_frac))
    idx = np.argsort(xv)[:k]
    X = np.log(xv[idx]); Y = np.log(yv[idx])
    A = np.vstack([np.ones_like(X), X]).T
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    slope = coef[1]; intercept = coef[0]
    return float(slope), float(intercept), float(np.exp(intercept))

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size for two samples (pooled SD)."""
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2: return np.nan
    m1, m2 = np.mean(a), np.mean(b)
    s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1-1)*s1*s1 + (n2-1)*s2*s2) / (n1+n2-2))
    if sp == 0: return np.nan
    return float((m1 - m2) / sp)

def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta (nonparametric effect size)."""
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0: return np.nan
    # Efficient approximate: sort and use ranks
    a_sorted = np.sort(a); b_sorted = np.sort(b)
    i = j = more = less = 0
    na, nb = len(a_sorted), len(b_sorted)
    while i < na and j < nb:
        if a_sorted[i] > b_sorted[j]:
            less += na - i; j += 1
        elif a_sorted[i] < b_sorted[j]:
            more += nb - j; i += 1
        else:
            # ties: advance both (approximate handling)
            i += 1; j += 1
    return float((more - less) / (na * nb))

def collapse_score(x_list: List[np.ndarray], y_list: List[np.ndarray]) -> float:
    """
    Simple 'collapse' score for sets of curves: concatenate normalized residuals after
    z-scoring each curve; return 1 - normalized SSE (ad hoc but useful to compare settings).
    """
    if len(x_list) < 2: return np.nan
    # Interpolate each curve onto a common x-grid (quantile grid of pooled x)
    pooled_x = np.concatenate([x for x in x_list if x.size > 0])
    if pooled_x.size == 0: return np.nan
    qx = np.quantile(pooled_x, np.linspace(0.05,0.95,50))
    curves = []
    for x,y in zip(x_list, y_list):
        if x.size == 0 or y.size == 0: continue
        # sort by x
        idx = np.argsort(x); x = x[idx]; y = y[idx]
        # select finite positive region
        m = np.isfinite(x) & np.isfinite(y) & (x>0) & (y>0)
        x = x[m]; y = y[m]
        if x.size < 5: continue
        yi = np.interp(qx, x, y, left=np.nan, right=np.nan)
        curves.append(yi)
    if len(curves) < 2: return np.nan
    C = np.vstack(curves)
    # z-score each curve ignoring nans
    C_mean = np.nanmean(C, axis=1, keepdims=True)
    C_std  = np.nanstd(C, axis=1, keepdims=True)
    Z = (C - C_mean) / (C_std + 1e-12)
    # variance across curves at each x
    var_across = np.nanvar(Z, axis=0)
    # higher score = tighter collapse (lower variance)
    score = 1.0 - float(np.nanmean(var_across) / (np.nanmean(np.nanvar(Z, axis=1)) + 1e-12))
    return score


# Core Experiment Runner

def run_one(config: Config, rng=None) -> Dict:
    """
    Execute one run (problem ensemble × algorithm × parameters).

    Returns
    -------
    dict with paths to CSV / JSON / PDF and a 'summary' payload.
    """
    if rng is None: rng = np.random.default_rng(0)
    outdir = config.outdir; outdir.mkdir(parents=True, exist_ok=True)

    # Build polynomial from ensemble
    if config.ensemble == "circle":
        roots = roots_on_circle(config.degree, 0.0, rng)
    elif config.ensemble == "jittered":
        roots = roots_on_circle(config.degree, jitter=0.15, rng=rng)
    elif config.ensemble == "clustered":
        roots = clustered_roots(config.degree, clusters=2, spread=0.02, rng=rng)
    elif config.ensemble == "multiple":
        roots = multiple_roots(config.degree, multiplicity=2, rng=rng)
    elif config.ensemble == "kac":
        roots = kac_like_roots(config.degree, rng=rng)
    else:
        raise ValueError("Unknown ensemble")

    f, df, d2f = poly_from_roots(roots)
    step = ALG_STEPS[config.algorithm]

    # Grid setup
    dtype = complex_dtype(config.precision)
    xs = np.linspace(config.xlim[0], config.xlim[1], config.grid, dtype=float)
    ys = np.linspace(config.ylim[0], config.ylim[1], config.grid, dtype=float)
    XX, YY = np.meshgrid(xs, ys)
    Z0 = (XX + 1j*YY).astype(dtype)
    dx = (config.xlim[1] - config.xlim[0]) / (config.grid - 1)
    dy = (config.ylim[1] - config.ylim[0]) / (config.grid - 1)

    # Allocate fields
    root_idx = -np.ones(Z0.shape, dtype=int)
    iters = np.zeros(Z0.shape, dtype=int)
    conv = np.zeros(Z0.shape, dtype=bool)
    p_field = np.full(Z0.shape, np.nan, dtype=float)
    r_last = np.full(Z0.shape, np.nan, dtype=float)
    r_fit = np.full(Z0.shape, np.nan, dtype=float)
    d0 = np.zeros(Z0.shape, dtype=float)       # nearest-root distance at start
    beta = np.full(Z0.shape, np.nan, dtype=float)  # α-proxies
    gamma = np.full(Z0.shape, np.nan, dtype=float)

    # Iterate per pixel (kept explicit for clarity & per-trajectory logging)
    for i in range(config.grid):
        for j in range(config.grid):
            z = Z0[i,j]
            # α-proxies at start (β ~ Newton step length; γ ~ curvature surrogate)
            fz = f(z); dfz = df(z); d2fz = d2f(z)
            if dfz != 0:
                beta[i,j] = abs(fz)/abs(dfz)
                gamma[i,j] = abs(d2fz)/(2.0*abs(dfz))
            # nearest-root distance at start
            d0[i,j] = float(np.min(np.abs(roots - z)))
            # trace iteration
            zs = [z]
            converged = False
            ridx = -1
            for k in range(config.max_iters):
                if df(z) == 0:
                    break
                z = step(z, f, df, d2f, lam=config.damping)
                zs.append(z)
                if abs(f(z)) < (1e-10 if dtype==np.complex128 else 1e-7):
                    dists = np.abs(roots - z)
                    jstar = int(np.argmin(dists))
                    if dists[jstar] < (1e-8 if dtype==np.complex128 else 1e-5):
                        converged = True; ridx = jstar; break
            conv[i,j] = converged
            root_idx[i,j] = ridx
            iters[i,j] = len(zs)-1
            if converged:
                zstar = roots[ridx]
                errors = [abs(zk - zstar) for zk in zs]
                p_hat, rL, rF = estimate_order_and_contraction(errors)
                p_field[i,j] = p_hat; r_last[i,j] = rL; r_fit[i,j] = rF

    # Boundary and distance-to-boundary
    B = boundary_mask(root_idx)
    boundary_density = float(B.mean())
    delta = distance_to_boundary(B, dx, dy)

    # Flatten dataset for analysis
    df_flat = pd.DataFrame({
        "x0": XX.ravel(), "y0": YY.ravel(),
        "converged": conv.ravel().astype(int),
        "root_idx": root_idx.ravel(),
        "iters": iters.ravel().astype(float),
        "p": p_field.ravel(), "r_last": r_last.ravel(), "r_fit": r_fit.ravel(),
        "d0": d0.ravel(), "delta": delta.ravel(),
        "beta": beta.ravel(), "gamma": gamma.ravel(),
        "is_boundary": B.ravel().astype(int),
    })

    # Basic summaries
    def basic_stats(a: np.ndarray) -> Dict[str,float]:
        a = a[np.isfinite(a)]
        if a.size == 0: return {"n":0,"mean":np.nan,"median":np.nan,"p90":np.nan}
        return {"n":int(a.size),"mean":float(np.mean(a)),"median":float(np.median(a)),"p90":float(np.percentile(a,90))}

    # Finite-size scaling slopes: (2-p) vs δ near boundary; r_fit vs δ near boundary
    slope_p, intercept_p, A_p = loglog_slope(df_flat["delta"].values, (2.0 - df_flat["p"].values))
    slope_r, intercept_r, A_r = loglog_slope(df_flat["delta"].values, df_flat["r_fit"].values)

    summary = {
        "ensemble": config.ensemble,
        "degree": int(config.degree),
        "algorithm": config.algorithm,
        "damping": float(config.damping),
        "precision": config.precision,
        "grid": int(config.grid),
        "boundary_density": boundary_density,
        "stats": {
            "iters": basic_stats(df_flat.loc[df_flat["converged"]==1, "iters"].values),
            "p": basic_stats(df_flat["p"].values),
            "r_last": basic_stats(df_flat["r_last"].values),
            "r_fit": basic_stats(df_flat["r_fit"].values),
            "delta": basic_stats(df_flat["delta"].values),
            "d0": basic_stats(df_flat["d0"].values),
            "beta": basic_stats(df_flat["beta"].values),
            "gamma": basic_stats(df_flat["gamma"].values),
        },
        "finite_size_scaling": {
            "slope_p_vs_delta_small": slope_p,
            "slope_rfit_vs_delta_small": slope_r
        }
    }

    # Save CSV & JSON
    tag = f"{config.ensemble}_n{config.degree}_{config.algorithm}_lam{config.damping}_{config.precision}_g{config.grid}"
    csv_path  = outdir / f"pixels_{tag}.csv"
    json_path = outdir / f"summary_{tag}.json"
    df_flat.to_csv(csv_path, index=False)
    with open(json_path, "w") as f: json.dump(summary, f, indent=2)

    # Report: Figures (one chart per figure)
    def imshow(field, title, xs, ys, outpath):
        plt.figure(figsize=(6,6), dpi=140)
        extent = [xs.min(), xs.max(), ys.min(), ys.max()]
        plt.imshow(field, origin="lower", extent=extent, aspect="equal")
        plt.xlabel("Re z0"); plt.ylabel("Im z0"); plt.title(title); plt.colorbar()
        plt.tight_layout(); plt.savefig(outpath, dpi=160); plt.close()
        return str(outpath)

    # Fields
    im1 = imshow(np.where(root_idx>=0, root_idx.astype(float), np.nan),
                 "Basin (root index)", xs, ys, outdir / f"field_basin_{tag}.png")
    im2 = imshow(iters.astype(float), "Iterations to converge", xs, ys, outdir / f"field_iters_{tag}.png")
    im3 = imshow(p_field, "Empirical order p", xs, ys, outdir / f"field_p_{tag}.png")
    im4 = imshow(r_fit,   "Contraction r_fit", xs, ys, outdir / f"field_rfit_{tag}.png")
    im5 = imshow(delta,   "Boundary distance δ", xs, ys, outdir / f"field_delta_{tag}.png")
    im6 = imshow(d0.astype(float), "Nearest-root distance d0", xs, ys, outdir / f"field_d0_{tag}.png")

    # ECDFs
    def plot_ecdf_to(path, values, title):
        v,y = ecdf(values)
        plt.figure(figsize=(6,4), dpi=140)
        if v.size>0: plt.plot(v,y)
        plt.xlabel("value"); plt.ylabel("ECDF"); plt.title(title)
        plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
        return str(path)

    ec1 = plot_ecdf_to(outdir/f"ecdf_iters_{tag}.png", df_flat.loc[df_flat["converged"]==1, "iters"].values, "ECDF — iterations")
    ec2 = plot_ecdf_to(outdir/f"ecdf_p_{tag}.png", df_flat["p"].values, "ECDF — p")
    ec3 = plot_ecdf_to(outdir/f"ecdf_rfit_{tag}.png", df_flat["r_fit"].values, "ECDF — r_fit")

    # Binned means with bootstrap CI
    def plot_binned_to(path, x, y, title, bins=30):
        m = np.isfinite(x) & np.isfinite(y)
        xv, yv = x[m], y[m]
        plt.figure(figsize=(6,4), dpi=140)
        if xv.size >= 10:
            qs = np.quantile(xv, np.linspace(0,1,bins+1))
            centers = 0.5*(qs[:-1] + qs[1:])
            means = []; los=[]; his=[]
            rng = np.random.default_rng(0)
            for a,b in zip(qs[:-1], qs[1:]):
                sel = (xv>=a)&(xv<b); yy = yv[sel]
                if yy.size == 0: means.append(np.nan); los.append(np.nan); his.append(np.nan); continue
                means.append(float(np.mean(yy)))
                lo,hi = bootstrap_ci(yy, np.mean, B=300, rng=rng)
                los.append(lo); his.append(hi)
            centers=np.array(centers); means=np.array(means); los=np.array(los); his=np.array(his)
            plt.plot(centers, means); plt.fill_between(centers, los, his, alpha=0.3)
        plt.xlabel("x"); plt.ylabel("binned mean"); plt.title(title)
        plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
        return str(path)

    bm1 = plot_binned_to(outdir/f"binned_delta_p_{tag}.png", df_flat["delta"].values, df_flat["p"].values, "δ → p (binned mean, 95% CI)")
    bm2 = plot_binned_to(outdir/f"binned_delta_rfit_{tag}.png", df_flat["delta"].values, df_flat["r_fit"].values, "δ → r_fit (binned mean, 95% CI)")
    bm3 = plot_binned_to(outdir/f"binned_d0_iters_{tag}.png", df_flat["d0"].values, df_flat["iters"].values, "d0 → iterations (binned mean, 95% CI)")

    # Scaling panels (log–log views)
    def plot_loglog_to(path, x, y, title):
        m = np.isfinite(x) & np.isfinite(y) & (x>0) & (y>0)
        xv, yv = x[m], y[m]
        plt.figure(figsize=(6,4), dpi=140)
        if xv.size > 20:
            plt.plot(np.log(xv), np.log(yv))
        plt.xlabel("log x"); plt.ylabel("log y"); plt.title(title)
        plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
        return str(path)

    ll1 = plot_loglog_to(outdir/f"loglog_2minusp_vs_delta_{tag}.png", df_flat["delta"].values, (2.0 - df_flat["p"].values), "log–log: (2−p) vs δ")
    ll2 = plot_loglog_to(outdir/f"loglog_rfit_vs_delta_{tag}.png", df_flat["delta"].values, df_flat["r_fit"].values, "log–log: r_fit vs δ")

    # Collate PDF report
    pdf_path = outdir / f"report_{tag}.pdf"
    with PdfPages(pdf_path) as pdf:
        for impath, title in [
            (im1, "Basin"), (im2, "Iterations"), (im3, "Empirical order p"),
            (im4, "Contraction r_fit"), (im5, "Boundary distance δ"), (im6, "Nearest-root distance d0"),
            (ec1, "ECDF iters"), (ec2, "ECDF p"), (ec3, "ECDF r_fit"),
            (bm1, "Binned δ→p"), (bm2, "Binned δ→r_fit"), (bm3, "Binned d0→iters"),
            (ll1, "Scaling (2−p) vs δ"), (ll2, "Scaling r_fit vs δ"),
        ]:
            if impath is None: continue
            fig = plt.figure(figsize=(8,6), dpi=140)
            img = plt.imread(impath)
            plt.imshow(img); plt.axis('off'); plt.title(title)
            pdf.savefig(fig); plt.close(fig)

        # Text page with summary (truncated for space)
        fig = plt.figure(figsize=(8.5, 11), dpi=140)
        txt = json.dumps(summary, indent=2)
        fig.text(0.02, 0.98, "Summary (truncated)", va="top", ha="left", fontsize=10)
        lines = txt.splitlines()[:60]
        fig.text(0.02, 0.95, "\n".join(lines), va="top", ha="left", fontsize=8)
        pdf.savefig(fig); plt.close(fig)

    return {"csv": str(csv_path), "json": str(json_path), "pdf": str(pdf_path), "summary": summary}


# Argument parsing and multi-run driver

def main():
    """
    CLI entry point. See --help for options.

    Example:
    --------
    python convergence_theory_experiment_documented.py \
        --grid 260 \
        --degrees 3 5 8 \
        --ensembles circle jittered clustered \
        --algs newton halley chebyshev \
        --lambdas 0.8 1.0 1.2 \
        --precisions complex128 \
        --outdir ./theory_outputs_doc
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=260, help="Grid resolution per axis")
    ap.add_argument("--degrees", type=int, nargs="+", default=[3,5,8], help="Polynomial degrees")
    ap.add_argument("--ensembles", type=str, nargs="+", default=["circle","jittered","clustered"], help="Problem ensembles")
    ap.add_argument("--algs", type=str, nargs="+", default=["newton","halley","chebyshev"], help="Algorithms to run")
    ap.add_argument("--lambdas", type=float, nargs="+", default=[1.0], help="Damping parameters")
    ap.add_argument("--precisions", type=str, nargs="+", default=["complex128"], help="Precisions: complex128 or complex64")
    ap.add_argument("--max_iters", type=int, default=80, help="Max iterations per start")
    ap.add_argument("--outdir", type=str, default="./theory_outputs_doc", help="Output directory")
    ap.add_argument("--xlim", type=float, nargs=2, default=[-2.0, 2.0], help="x-range of grid")
    ap.add_argument("--ylim", type=float, nargs=2, default=[-2.0, 2.0], help="y-range of grid")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    manifest = {"runs": []}
    for ens in args.ensembles:
        for n in args.degrees:
            for alg in args.algs:
                for lam in args.lambdas:
                    for prec in args.precisions:
                        cfg = Config(
                            ensemble=ens, degree=n, damping=lam, precision=prec,
                            grid=args.grid, xlim=tuple(args.xlim), ylim=tuple(args.ylim),
                            max_iters=args.max_iters, algorithm=alg, outdir=outdir
                        )
                        res = run_one(cfg)
                        manifest["runs"].append({
                            "csv": res["csv"], "json": res["json"], "pdf": res["pdf"],
                            "ensemble": ens, "degree": n, "algorithm": alg,
                            "lambda": lam, "precision": prec
                        })

    with open(outdir/"manifest.json","w") as f:
        json.dump(manifest, f, indent=2)
    print("Done. Reports in", outdir.resolve())


if __name__ == "__main__":
    main()
