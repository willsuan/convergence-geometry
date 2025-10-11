# Convergence Geometry — Experimental Program

This package implements an experiment suite to study **algorithmic properties** of iterative solvers via **convergence fields**, aiming for **scaling laws** and **universality** that complement Newton–Kantorovich and complex-dynamical theory.

## Motivation (theory)
- **Newton–Kantorovich & α-theory:** provide sufficient (local) convergence conditions. We map empirical order \(p\) and contractivity \(r\) outside guaranteed regions and calibrate α-like proxies.
- **Complex dynamics:** Newton maps are rational; basins/Julia sets are understood qualitatively. We add **quantitative fields** and **finite-size scaling** near basin boundaries.
- **Conditioning & sensitivity:** link geometric covariates (distance to boundary \(\delta\), nearest-root distance \(d_0\), crowding) to observed rates and stability.

## What this code does
- **Problem ensembles:** `circle`, `jittered`, `clustered`, `multiple`, `kac` (synthetic), with degree \(n\).
- **Algorithms:** `newton`, `halley`, `chebyshev` (cubic conv.).
- **Metrics per pixel:** basin, iterations, **empirical order \(p\)**, **contraction \(r\)** (last, fit), boundary distance \(\delta\), nearest-root distance \(d_0\), α-proxies \(β=|f|/|f'|\), \(γ≈|f''|/(2|f'|)\).
- **Statistics:** ECDFs; **binned means with bootstrap CIs**; **finite-size scaling** via log–log slope fits near small \(\delta\).
- **Reports:** one **PDF per run** collating fields, ECDFs, binned-CI plots, log–log panels, plus JSON summaries and per-pixel CSVs.

## Quick start
```bash
python convergence_theory_experiment.py --grid 300 --degrees 3 5 8 \
  --ensembles circle jittered clustered \
  --algs newton halley chebyshev \
  --lambdas 0.8 1.0 1.2 \
  --precisions complex128
```

Artifacts are written to `./theory_outputs/`:
- `pixels_*.csv` — per-pixel dataset
- `summary_*.json` — numerical summary + finite-size slopes
- `report_*.pdf` — multi-page report for the run
- `manifest.json` — index of all run artifacts

## Extending
- Add more algorithms to `ALG_STEPS` (e.g., Broyden, Anderson).
- Add crowding metrics (k-NN root distances) and include multivariate regressions.
- Swap synthetic `kac` generator for true Kac polynomials by coefficient sampling + numerical root-finding (if acceptable).
- Incorporate mixed-precision & iterative refinement loops as separate axes.

## What to test
- **Boundary-law:** slope of \((2-p)\) vs \(\delta\) on log–log axes; check stability across ensembles/algorithms/degree.
- **Precision line:** repeat with `complex64` and look for a sharp degradation curve in \(p\) as \(\delta\) decreases.
- **Universality:** attempt finite-size **collapse** by rescaling \(\delta\) by degree-dependent factors.
