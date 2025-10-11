#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finite-Size Collapse Utility for Convergence Geometry

Given a manifest of runs across **degrees** (and possibly algorithms/ensembles),
this tool tries to rescale the boundary distance δ by a factor of n^α (or a more
general n^α c^β if you pass --scale_by 'degree' and --scale_by2 another numeric
run attribute) to **collapse** the curves of y(x) across n onto a master curve.

By default we collapse y = E[2 − p | δ] (binned means) as a function of δ.
You can also choose r_fit or iters.

It grid-searches α over a range and returns the α that maximizes a collapse score
(variance reduction across curves after z-scoring), and produces a PDF with pre-
and post-collapse plots.

Usage:
  python finite_size_collapse.py --manifest ./theory_outputs_doc/manifest.json --y 2minusp --bins 40
  python finite_size_collapse.py --manifest ./theory_outputs_doc/manifest.json --y r_fit --bins 50

Plot policy: matplotlib only, one chart per figure, no explicit colors.
"""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def binned_xy(delta, y, bins=40):
    m = np.isfinite(delta) & np.isfinite(y)
    x = delta[m]; yy = y[m]
    if x.size < 10: return np.array([]), np.array([])
    qs = np.quantile(x, np.linspace(0,1,bins+1))
    centers = 0.5*(qs[:-1] + qs[1:])
    means = []
    for a,b in zip(qs[:-1], qs[1:]):
        sel = (x>=a)&(x<b); yb = yy[sel]
        means.append(np.mean(yb) if yb.size>0 else np.nan)
    centers = np.array(centers); means = np.array(means)
    m2 = np.isfinite(centers) & np.isfinite(means)
    return centers[m2], means[m2]

def collapse_score(curves_x, curves_y):
    # Interpolate onto a common x-grid and compute variance across curves after z-scoring each curve
    pooled = np.concatenate([x for x in curves_x if x.size>0])
    if pooled.size == 0: return np.nan
    grid = np.quantile(pooled, np.linspace(0.05,0.95,60))
    Ys = []
    for x,y in zip(curves_x, curves_y):
        if x.size==0 or y.size==0: continue
        idx = np.argsort(x); x = x[idx]; y = y[idx]
        m = np.isfinite(x) & np.isfinite(y) & (x>0) & (y>0)
        x = x[m]; y = y[m]
        if x.size < 5: continue
        yi = np.interp(grid, x, y, left=np.nan, right=np.nan)
        Ys.append(yi)
    if len(Ys) < 2: return np.nan
    Y = np.vstack(Ys)
    # z-score rows
    mu = np.nanmean(Y, axis=1, keepdims=True)
    sd = np.nanstd(Y, axis=1, keepdims=True) + 1e-12
    Z = (Y - mu)/sd
    var_across = np.nanvar(Z, axis=0)
    # lower variance => better collapse; map to [0,1] score
    return float(1.0 - np.nanmean(var_across))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--y", type=str, default="2minusp", choices=["2minusp","r_fit","iters"])
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--alpha_min", type=float, default=-2.0)
    ap.add_argument("--alpha_max", type=float, default= 2.0)
    ap.add_argument("--alpha_steps", type=int, default=81)
    ap.add_argument("--outdir", type=str, default="./collapse")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, "r") as f:
        mani = json.load(f)
    runs = mani.get("runs", [])

    # Group by degree (within a fixed (ensemble, algorithm, precision, lambda) ideally)
    # For simplicity here, we group by 'degree' and ignore other axes; you can pre-filter manifest if needed.
    # Build curves per degree from per-pixel CSVs (binned means).
    curves_by_deg = {}
    for r in runs:
        deg = r.get("degree")
        csv_path = Path(r["csv"])
        if not csv_path.exists(): continue
        df = pd.read_csv(csv_path, usecols=["delta","p","r_fit","iters"])
        if args.y == "2minusp":
            yvals = 2.0 - df["p"].values
        elif args.y == "r_fit":
            yvals = df["r_fit"].values
        else:
            yvals = df["iters"].values
        xvals = df["delta"].values
        x, y = binned_xy(xvals, yvals, bins=args.bins)
        if x.size>0 and y.size>0:
            curves_by_deg.setdefault(deg, []).append((x,y))

    # Pre-collapse plots (raw)
    plt.figure(figsize=(6,4), dpi=140)
    for deg, curves in sorted(curves_by_deg.items()):
        for x,y in curves:
            if x.size>0 and y.size>0:
                plt.plot(x, y)
    plt.xlabel("δ"); plt.ylabel(args.y); plt.title("Pre-collapse curves (binned means)")
    pre_png = outdir / f"precollapse_{args.y}.png"
    plt.tight_layout(); plt.savefig(pre_png, dpi=160); plt.close()

    # Grid search alpha for x' = x * degree^alpha
    alphas = np.linspace(args.alpha_min, args.alpha_max, args.alpha_steps)
    best_alpha, best_score = None, -np.inf
    for a in alphas:
        curves_x = []; curves_y = []
        for deg, curves in curves_by_deg.items():
            scale = (deg ** a) if (deg is not None) else 1.0
            for x,y in curves:
                curves_x.append(x * scale)
                curves_y.append(y)
        score = collapse_score(curves_x, curves_y)
        if np.isfinite(score) and score > best_score:
            best_score, best_alpha = score, a

    # Post-collapse plots with best alpha
    plt.figure(figsize=(6,4), dpi=140)
    for deg, curves in sorted(curves_by_deg.items()):
        scale = (deg ** best_alpha) if (deg is not None) else 1.0
        for x,y in curves:
            if x.size>0 and y.size>0:
                plt.plot(x * scale, y)
    plt.xlabel("δ * n^alpha"); plt.ylabel(args.y); plt.title(f"Post-collapse (alpha≈{best_alpha:.3f})")
    post_png = outdir / f"postcollapse_{args.y}.png"
    plt.tight_layout(); plt.savefig(post_png, dpi=160); plt.close()

    # Save a small PDF report
    from matplotlib.backends.backend_pdf import PdfPages
    pdf_path = outdir / f"collapse_{args.y}.pdf"
    with PdfPages(pdf_path) as pdf:
        for im, title in [(pre_png, "Pre-collapse"), (post_png, f"Post-collapse (alpha≈{best_alpha:.3f})")]:
            fig = plt.figure(figsize=(8,6), dpi=140)
            img = plt.imread(im); plt.imshow(img); plt.axis('off'); plt.title(title)
            pdf.savefig(fig); plt.close(fig)
        # text page
        fig = plt.figure(figsize=(8.5, 11), dpi=140)
        fig.text(0.02, 0.98, "Best collapse exponent", va="top", ha="left", fontsize=10)
        fig.text(0.02, 0.94, f"alpha ≈ {best_alpha:.6f}\nscore ≈ {best_score:.6f}", va="top", ha="left", fontsize=10)
        pdf.savefig(fig); plt.close(fig)

    print("Best alpha:", best_alpha, "score:", best_score)
    print("Wrote:", pre_png)
    print("Wrote:", post_png)
    print("Wrote:", pdf_path)

if __name__ == "__main__":
    main()
