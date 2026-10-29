# Reproducing the experiments

## Installation and fixed numerical policy

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,performance]'
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8
export HF_FAS_GS_BACKEND=numba
python -m pytest -q
```

`performance` installs optional Numba. For a reference Python GS implementation, omit that extra and set `HF_FAS_GS_BACKEND=python`; wall times will differ. `requirements-ci.txt` fixes the NumPy/SciPy/Matplotlib/pytest versions used by CI. Performance runs use eight FFT workers, one BLAS thread, and a warmed GS backend; setup and warmup are excluded from solver time.

The paper's eight systems are `LiH Li2 BeHplus BeH2 BH H2O N2 CH4`. Positions and charges are defined in `hf_fas/fem/molecule.py`, with methane geometries in `hf_fas/experiments/starts.py`. All distances are in bohr. Defaults are the paper box `[-8,8]^3`, four base cells per axis, nuclear quadrature order 8, two voxels per cell edge, and DCT-VGF precomputation. Level L means L halvings of the base mesh:

| Level | Cells per axis | h (bohr) | Orbital degrees of freedom |
|---|---:|---:|---:|
| L3 | 32 | 0.5 | 29,791 |
| L4 | 64 | 0.25 | 250,047 |
| L5 | 128 | 0.125 | 2,048,383 |

Each FAS cycle uses component correction, 2+2 smoothing, and forced joint KKT Newton at the root. The FAS-H response trigger is 0.4, with at most one horizontal Newton outer step, 60 GMRES iterations, restart 60, base tolerance `1e-8`, and forcing ceiling `1e-2`. Pure horizontal Newton allows 100 outer steps and 160 GMRES iterations, restart 100, forcing ceiling `1e-3`, and disables GS fallback. SCF uses DIIS history 6, damping 1 and inner factor 0.05. All complete comparison solves target residual `1e-8`.

## Shared inputs, preparation and full comparisons

A complete example, with preparation explicitly separated from timing:

```bash
python -m hf_fas.experiments.reproduce --system LiH --level 3 \
  --stage prepare --output results/LiH-L3
python -m hf_fas.experiments.reproduce --system LiH --level 3 \
  --stage solve --output results/LiH-L3
```

Alternatively omit `--stage` to perform both stages. Coarse-to-fine preparation solves lower grids and only prolongates/selects the finest initial frame; it saves `initial.npz` before computing the reference. The default continuation is FAS for LiH, BeHplus and BeH2, and SCF for the other systems, matching the canonical campaign. `--continuation fas|scf` makes this choice explicit. The independently computed reference uses SCF to `1e-10`, at most 300 iterations. `--reference-method Newton-horizontal` permits strict reference preparation with horizontal Newton if SCF does not converge; record this change when reporting results. Preparation failure is retained and stops the run.

The three comparison solvers receive copies of the same saved initial state and reference. SHA256 checks reject changed inputs. `--methods FAS-H SCF` selects a subset. Existing saved inputs can be reused with `--inputs /path/to/directory`, containing `initial.npz`, `reference.npz`, `ready.json` and `initialization.json`; this also accepts the original experiment input format when its backend fingerprint matches. Use matching grid, geometry, workers and kernel options. New method outputs refuse to overwrite an existing method directory.

For the grid convergence and solver-cost experiments, repeat for the eight systems at levels 3, 4 and 5. For example, after allocating suitable compute resources:

```bash
for level in 3 4 5; do
  for system in LiH Li2 BeHplus BeH2 BH H2O N2 CH4; do
    python -m hf_fas.experiments.reproduce --system "$system" --level "$level" \
      --output "results/${system}-L${level}"
  done
done
```

L4/L5 are substantial calculations: the original campaign used 8 CPUs and up to 256 GiB per L5 allocation, with individual solves taking hours. Pair fields and independent validation consume additional memory. Run those commands on an allocated compute node. The runner itself has no Slurm dependency; a Slurm example is:

```bash
srun --cpus-per-task=8 --mem=256G --time=24:00:00 \
  python -m hf_fas.experiments.reproduce --system CH4 --level 5 \
  --output results/CH4-L5
```

These are fresh runs using the preserved numerical algorithms. They regenerate starts and references rather than downloading historical checkpoint arrays; bitwise input identity, exact iteration counts and original wall times are not guaranteed. Retain the generated source hashes, runtime versions and inputs when comparing runs. The historical output snapshots allow checking the paper's reported counts without rerunning every expensive experiment.

## Outputs, checks and cost accounting

Each method directory contains `invocation.json`, accepted `history.jsonl`, `selected.npz` and `outcome.json`; `summary.json` retains every attempt, including unqualified endpoints and exceptions. `run-*.json` records setup, versions, host, CPU affinity, Slurm ID when present, source hashes and thread settings. Input/reference hashes are recorded in each method invocation.

An endpoint qualifies only if the full residual is below `1e-8`, electronic-energy error below `1e-7` Ha, occupied-projector and relative-density errors below `1e-6`, and electron-count error below `1e-9`. Separately constructed streamed inverse-FFT VGF weights must give raw and rebuilt residuals below `1e-8` and an energy difference below `1e-9` Ha. Agreement establishes a common discrete stationary endpoint; it is not a ground-state or continuum-accuracy certificate.

`numerical_seconds` excludes accepted-history callback time; setup, common initialization, reference generation, endpoint checks and final persistence are outside the solver timer. Work counters are frozen before validation. `work_summary.fft_work` sums the packed scalar-source counts at each level weighted by padded FFT `P log2(P)` and normalized to the finest convolution, including triggered FAS-H response work. Scalar FFT counts alone do not express multilevel cost.

For timing comparisons, use the same saved inputs, node, CPU affinity, eight FFT workers and one BLAS thread, and run the methods sequentially in one allocation. Repeat with new output directories and `--inputs` to reuse the exact start/reference. Absolute times depend on hardware and runtime; iteration counts alone are not speedups.

**Recorded L5 timing scope:** the original 24 L5 solves ran in separate allocations, including different nodes. The retained cost-ratio data uses eight same-node L5 panels: where necessary, audited full reruns replace the original FAS-H/Newton times while retaining the exact start/reference arrays. These include the Li2 L5 rerun completed after the earlier 22-panel snapshot and the qualified BH L5 FAS-H/Newton reruns on bigMem4, paired with the historical SCF on that node. All eight L5 systems appear in the 24 retained comparison panels. The selected Li2/BH inputs, endpoint checks, source hashes and costs are retained in `added_panel_evidence` within `data/vgf-cost-ratios-current.json`. Same-node measurements in different allocations are not simultaneous controlled timing repeats. Keep node, affinity and allocation provenance when interpreting these ratios. Fresh paired timing runs should use one allocation as described above.

## Multigrid mechanism

After preparation, replay the first FAS-H cycle:

```bash
python -m hf_fas.experiments.reproduce --system LiH --level 4 \
  --stage phases --output results/LiH-L4
```

Repeat for the eight L4 systems to collect the frequency-band measurements. `phases/result.json` reports the aligned, horizontally projected orbital error in consistent-mass sine bands at the initial state, after pre-smoothing, after coarse correction (`post_*-raw`), and after post-smoothing (`post_*`). Low modes have `max(k) <= N/4`; high modes have `max(k) > N/2`. Divide the square root of each band energy by the initial total-error norm for plotting. Phase states precede the optional physical Newton correction; that correction's records and final residual are reported separately. Regression tests check that phase observation preserves solver states and all non-timing counters exactly.

## Supplementary cases

`data/experiment-matrix.json` records the 42 prescribed panels, initialization families and original per-method budgets. The portable runner supports their numerical ingredients:

```bash
# Two methane geometry changes (run each with a distinct output directory).
python -m hf_fas.experiments.reproduce --system CH4 --level 3 \
  --geometry rotated --output results/CH4-rotated-L3
python -m hf_fas.experiments.reproduce --system CH4 --level 3 \
  --geometry distorted --output results/CH4-distorted-L3

# Initial-state study: for each of LiH, H2O, CH4, compare all four starts.
python -m hf_fas.experiments.reproduce --system LiH --level 3 \
  --start core-perturbed --output results/LiH-core-perturbed-L3

# Original versus signed density-compressed SCF (cutoff 1e-12).
python -m hf_fas.experiments.reproduce --system BH --level 3 \
  --methods SCF SCF-compressed --output results/BH-L3-compression
```

Starts are `continuation`, `continuation-perturbed`, `core`, `core-perturbed`. The preserved fixed seeds, low-sine perturbations and amplitudes are in `starts.py`; the reference is not used to build a start. Strict reference preparation always starts from the canonical continuation, even when the comparison start is a prescribed variant. For exact reference identity across all four panels, prepare the canonical inputs once and use `--base-inputs`:

```bash
for start in continuation continuation-perturbed core core-perturbed; do
  python -m hf_fas.experiments.reproduce --system LiH --level 3 \
    --base-inputs results/LiH-L3/inputs --start "$start" \
    --output "results/LiH-L3-${start}"
done
```

`--base-inputs` verifies the canonical saved input hashes and backend, constructs the selected start without accessing reference orbitals, and copies the exact canonical reference. Repeat this recipe for H2O and CH4. For declared repeats, use `--inputs` and a new output directory to reuse both exact arrays.

Original L3 core-start pure-Newton budgets were 1800 seconds; the portable runner has iteration caps but no internal wall-time cutoff. Apply the recorded budgets with the scheduler or `timeout` when reproducing budget censoring. Censoring outcomes and quota failures cannot be regenerated solely by changing solver iterations. All 114 recorded attempts, all 12 compression pairs, six local probes and historical L5 infrastructure outcomes remain in the snapshot; none is converted into successful fresh output. Historical reference-recovery and quota-management scripts have been removed.

## Recorded experiment data

The `data/` directory contains the experimental matrix, recorded outcomes, first-cycle frequency-band measurements, and cost-ratio data. `vgf-current-results.json` includes all 114 primary/supplementary attempts, their qualification status and the 24 additional L5 outcomes. `vgf-cost-ratios-current.json` preserves the selected same-node comparisons and the input/endpoint provenance of the Li2 and BH L5 reruns. `evidence-sha256.json` records hashes of these four data files. CI checks their integrity, endpoint qualifications and matched identities, alongside actual small-grid numerical tests.

Historical arrays and retired experiment packages are not distributed. Source paths recorded in the data are provenance, not requirements for running the code.
