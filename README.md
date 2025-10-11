# Convergence Geometry — Experimental Program

This package implements an experiment suite to study algorithmic properties of iterative solvers via convergence fields, aiming for scaling laws and universality that complement Newton–Kantorovich and complex-dynamical theory.

## Motivation (theory)
- Newton–Kantorovich & α-theory: provide sufficient (local) convergence conditions. We map empirical order \(p\) and contractivity \(r\) outside guaranteed regions and calibrate α-like proxies.
- Complex dynamics: Newton maps are rational; basins/Julia sets are understood qualitatively. We add quantitative fields and finite-size scaling near basin boundaries.
- Conditioning & sensitivity: link geometric covariates (distance to boundary \(\delta\), nearest-root distance \(d_0\), crowding) to observed rates and stability.

## What this code does
- Problem ensembles: `circle`, `jittered`, `clustered`, `multiple`, `kac` (synthetic), with degree \(n\).
- Algorithms: `newton`, `halley`, `chebyshev` (cubic conv.).
- Metrics per pixel: basin, iterations, empirical order \(p\), contraction \(r\) (last, fit), boundary distance \(\delta\), nearest-root distance \(d_0\), α-proxies \(β=|f|/|f'|\), \(γ≈|f''|/(2|f'|)\).
- Statistics: ECDFs; binned means with bootstrap CIs; finite-size scaling via log–log slope fits near small \(\delta\).
- Reports: one PDF per run collating fields, ECDFs, binned-CI plots, log–log panels, plus JSON summaries and per-pixel CSVs.

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
- Boundary-law: slope of \((2-p)\) vs \(\delta\) on log–log axes; check stability across ensembles/algorithms/degree.
- Precision line: repeat with `complex64` and look for a sharp degradation curve in \(p\) as \(\delta\) decreases.
- Universality: attempt finite-size collapse by rescaling \(\delta\) by degree-dependent factors.

## Sample Images
- Degree 3, circle ensemble, Newton, λ=1.0, fp64 precision, 120×120 grid, 60 iterations

<img width="360" height="360" alt="field_p_circle_n3_newton_lam1 0_complex128_g120" src="https://github.com/user-attachments/assets/36a32d1f-5fa2-4f96-9d97-d4b58384a4ca" />
<img width="360" height="360" alt="field_rfit_circle_n3_newton_lam1 0_complex128_g120" src="https://github.com/user-attachments/assets/8467bdea-81e5-405f-9f86-39ffe13d4151" />

<img width="360" height="360" alt="field_delta_circle_n3_newton_lam1 0_complex128_g120" src="https://github.com/user-attachments/assets/11095456-50ea-4019-b72f-bba9310de210" />
<img width="360" height="360" alt="field_d0_circle_n3_newton_lam1 0_complex128_g120" src="https://github.com/user-attachments/assets/895101f6-f70c-4864-9d64-0919754b7e62" />

<img width="360" height="360" alt="field_basin_circle_n3_newton_lam1 0_complex128_g120" src="https://github.com/user-attachments/assets/406f08f1-1a80-4060-9ca1-72dc86907a5d" />
<img width="360" height="360" alt="field_iters_circle_n3_newton_lam1 0_complex128_g120" src="https://github.com/user-attachments/assets/b29e7456-e93a-4d28-bff5-48edaa4911b5" />

<img width="360" height="640" alt="ecdf_iters_circle_n3_newton_lam1 0_complex128_g120" src="https://github.com/user-attachments/assets/bafe86b8-a013-4a08-b9b0-f610a5d87ed1" />
<img width="360" height="640" alt="ecdf_p_circle_n3_newton_lam1 0_complex128_g120" src="https://github.com/user-attachments/assets/7b4f459d-997c-4b53-816c-18275f9177de" />

<img width="360" height="640" alt="ecdf_rfit_circle_n3_newton_lam1 0_complex128_g120" src="https://github.com/user-attachments/assets/86043fec-7d72-4fcb-bb6a-a6d86746520d" />
<img width="360" height="640" alt="binned_delta_p_circle_n3_newton_lam1 0_complex128_g120" src="https://github.com/user-attachments/assets/73c9eae0-4ef1-4819-8fdb-0a735ac5c5b4" />

<img width="360" height="640" alt="binned_delta_rfit_circle_n3_newton_lam1 0_complex128_g120" src="https://github.com/user-attachments/assets/43f732b6-5fcb-4534-b50a-3e89f6139867" />
<img width="360" height="640" alt="binned_d0_iters_circle_n3_newton_lam1 0_complex128_g120" src="https://github.com/user-attachments/assets/7530a908-dad8-40ed-bb2b-c4aa3767bf3c" />

