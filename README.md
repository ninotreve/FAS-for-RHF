# FAS for RHF

[![Numerical regression](https://github.com/ninotreve/FAS-for-RHF/actions/workflows/regression.yml/badge.svg)](https://github.com/ninotreve/FAS-for-RHF/actions/workflows/regression.yml)

Numerical implementation and experiment reproduction code for structure-preserving FAS applied to real-space restricted Hartree–Fock equations.

The implementation uses nested Q1 finite elements, consistent mass matrices, exact product projection, free-space VGF Coulomb convolution, and complete forced FAS coarse equations. The paper's **FAS-H** is the original `C-response` algorithm: a component FAS cycle followed, when needed, by one bounded physical horizontal Newton correction. The comparison methods are horizontal Newton and commutator-DIIS SCF.

Python 3.10 or newer and Linux are supported. The computational dependencies are NumPy and SciPy; Numba is an optional accelerator.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q

# Small real solve for checking the installation (not a paper-size experiment).
python -m hf_fas.experiments.reproduce \
  --system He --level 1 --half-width 4 --workers 1 \
  --output results/he-smoke
```

The smoke run prepares a shared initial state and independent reference, runs all three solvers, and checks each endpoint with a separately constructed inverse-FFT kernel. It exits with an error if any solver or check fails. Use a new output directory for each run.

See [REPRODUCING.md](REPRODUCING.md) for full-size experiments, grid refinement, performance measurements, mechanism probes, and the recorded data.

- `hf_fas/fem/`: molecular equations, constraints, FAS, horizontal Newton, SCF, initialization.
- `hf_fas/tensor/`: Q1 spaces, tensor inverses, free-space Coulomb and VGF kernels.
- `hf_fas/experiments/`: portable experiment runner, FAS-H policy, endpoint checks and phase diagnostics.
- `data/`: recorded experiment matrix, outcomes, and cost-ratio evidence.
- `tests/`: numerical regression, independent small-grid oracles and end-to-end checks.

GitHub Actions runs the regression suite on Python 3.10 and 3.12 for pushes and pull requests. Full L3–L5 production runs are described separately because of their computational cost.
