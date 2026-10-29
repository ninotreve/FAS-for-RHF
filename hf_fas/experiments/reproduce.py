"""Portable reproduction of the fixed Q1/VGF paper experiments."""
import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from hf_fas.fem import (FEMHierarchy, FASConfig, InitializationConfig,
    MolecularFEMModel, MolecularResidual, SCFConfig, initialize_hierarchy,
    solve_molecular_fas, solve_molecular_scf, solve_molecular_quotient, gs_scan)
from hf_fas.fem.coarse import CoarseConfig
from hf_fas.fem.molecule import Molecule, paper_molecules
from hf_fas.fem.records import model_work, work_difference
from . import fem_paper as paper
from .fem_run import save_state, write
from .fft_work import work_summary
from .mechanism import error_bands
from .q1_boundary_repairs import POLICY, RepairHierarchy
from .q1_fas_h_phase_probe import ObservedRepairHierarchy
from .starts import methane_centers, construct_start

SYSTEMS = ('LiH', 'Li2', 'BeHplus', 'BeH2', 'BH', 'H2O', 'N2', 'CH4')
METHODS = ('FAS-H', 'Newton-horizontal', 'SCF', 'SCF-compressed')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def build(args):
    spec = SimpleNamespace(elements=4, half_width=args.half_width,
        refinements=args.level, nuclear_order=8, fft_workers=args.workers,
        kernel_cache=args.cache, warp=0.)
    spaces, maps, setup = paper.build_tensor(spec)
    mol = (Molecule(np.array([6, 1, 1, 1, 1]), methane_centers(args.geometry))
           if args.system == 'CH4' else paper_molecules()[args.system])
    return [MolecularFEMModel(s, mol, c) for s, c in zip(spaces, maps)], setup


def prepare(args, models, origin):
    origin.mkdir(parents=True, exist_ok=False)
    if args.base_inputs:
        base, reference, ready = load_inputs(args.base_inputs, models[-1])
        save_state(origin/'initial.npz', construct_start(models[-1], base, args.start))
        shutil.copy2(args.base_inputs/'reference.npz', origin/'reference.npz')
        ready.update(initial_sha256=digest(origin/'initial.npz'), start=args.start)
        write(origin/'ready.json', ready)
        write(origin/'initialization.json', dict(finest_solved=False,
            reference_used_to_construct_start=False,
            parent_initial_sha256=read(args.base_inputs/'ready.json')['initial_sha256']))
        return
    method = args.continuation or ('fas' if args.system in ('LiH', 'BeHplus', 'BeH2') else 'scf')
    initial, record = initialize_hierarchy(FEMHierarchy(models),
        InitializationConfig(continuation_method=method,
            scf=SCFConfig(tolerance=1e-9)))
    write(origin/'initialization.json', record)
    if initial is None:
        raise RuntimeError('Coarse-to-fine initialization failed; see initialization.json')
    # The finest comparison start is saved BEFORE constructing the reference.
    continuation = initial.copy()
    initial = construct_start(models[-1], initial, args.start)
    save_state(origin/'initial.npz', initial)
    fine = models[-1]
    if args.reference_method == 'SCF':
        result = solve_molecular_scf(fine, continuation, SCFConfig(tolerance=1e-10, max_iterations=300))
    else:
        result = solve_molecular_quotient(fine, continuation,
            CoarseConfig(tolerance=1e-10, max_iterations=100,
                         krylov_iterations=160, gs_fallback=False))
    write(origin/'reference.json', dict(converged=result.converged, history=result.history,
        seconds=result.elapsed_seconds, method=args.reference_method))
    if not result.converged:
        raise RuntimeError('Strict reference did not converge; results retained')
    save_state(origin/'reference.npz', result.state)
    precision = paper.precision_endpoint_check(fine, result.state)
    write(origin/'reference-precision.json', precision)
    if not precision['qualified']:
        raise RuntimeError('Independent reference check failed')
    write(origin/'ready.json', dict(qualified=True, system=args.system,
        level=args.level, half_width=args.half_width, geometry=args.geometry,
        start=args.start, coulomb=fine.coulomb.fingerprint,
        initial_sha256=digest(origin/'initial.npz'),
        reference_sha256=digest(origin/'reference.npz'),
        initialization_method=method, finest_solved=False))


def load_inputs(origin, fine):
    ready = read(origin/'ready.json')
    if not ready['qualified'] or ready['coulomb'] != fine.coulomb.fingerprint:
        raise ValueError('Input backend differs or reference is unqualified')
    init = read(origin/'initialization.json')
    if init.get('finest_solved') is not False:
        raise ValueError('Comparison requires an unsolved finest-grid start')
    for label in ('initial', 'reference'):
        if digest(origin/(label+'.npz')) != ready[label+'_sha256']:
            raise ValueError('Input SHA256 differs: '+label)
    return paper.load_state(origin/'initial.npz', fine), paper.load_state(origin/'reference.npz', fine), ready


def solve(args, models, initial, reference, ready):
    outcomes = []
    cfg = FASConfig(correction='component', max_cycles=200)
    newton = CoarseConfig(tolerance=1e-8, max_iterations=100,
                         krylov_iterations=160, gs_fallback=False)
    for method in args.methods:
        directory = args.output/method
        directory.mkdir(parents=True, exist_ok=False)
        hierarchy = RepairHierarchy(models, 'C-response')
        callback_seconds = 0.
        def callback(state, row):
            nonlocal callback_seconds
            tick = perf_counter()
            with (directory/'history.jsonl').open('a') as f:
                f.write(json.dumps(row, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item(), allow_nan=False)+'\n')
            callback_seconds += perf_counter()-tick
        scf = SCFConfig(exchange_compression_tolerance=1e-12 if method == 'SCF-compressed' else None)
        write(directory/'invocation.json', dict(method=method, FAS_config=asdict(cfg),
            Newton_config=asdict(newton), SCF_config=asdict(scf), response_policy=POLICY,
            initial_sha256=ready['initial_sha256'], reference_sha256=ready['reference_sha256']))
        before = [model_work(m) for m in models]
        try:
            # Each solver receives its own copy of the identical saved start.
            if method == 'FAS-H':
                result = solve_molecular_fas(hierarchy, initial.copy(), cfg, callback)
            elif method.startswith('SCF'):
                result = solve_molecular_scf(models[-1], initial.copy(), scf, callback)
            else:
                result = solve_molecular_quotient(models[-1], initial.copy(), newton, callback)
            # Freeze solve work before persistence or independent validation.
            counts = [work_difference(m, b) for m, b in zip(models, before)]
            save_state(directory/'selected.npz', result.state)
            endpoint = paper.endpoint_check(models[-1], result.state, reference)
            precision = paper.precision_endpoint_check(models[-1], result.state) if result.converged else None
            qualified = bool(result.converged and endpoint['qualified'] and precision and precision['qualified'])
            row = dict(method=method, qualified=qualified, converged=result.converged,
                status='qualified' if qualified else 'unqualified',
                iterations=len(result.history)-1, final_residual=result.final_residual,
                application_seconds=result.elapsed_seconds, callback_seconds=callback_seconds,
                numerical_seconds=result.elapsed_seconds-callback_seconds,
                history=result.history, work=counts, work_summary=work_summary(models, counts),
                endpoint=endpoint, precision_validation=precision,
                repair_records=hierarchy.repair_records,
                selected_sha256=digest(directory/'selected.npz'),
                initial_sha256=ready['initial_sha256'], reference_sha256=ready['reference_sha256'])
        except Exception as exc:
            row = dict(method=method, qualified=False, status='failed', reason=str(exc),
                work=[work_difference(m, b) for m, b in zip(models, before)])
        write(directory/'outcome.json', row)
        outcomes.append(row)
        write(args.output/'summary.json', outcomes)
        print(method, row['status'], flush=True)
    return all(row['qualified'] for row in outcomes)


def phases(args, models, initial, reference):
    directory = args.output/'phases'
    directory.mkdir(parents=True, exist_ok=False)
    fine = models[-1]
    hierarchy = ObservedRepairHierarchy(models, 'C-response')
    hierarchy.current_cycle = 1
    residual = fine.residual(initial)
    before = [model_work(m) for m in models]
    final, defect = hierarchy.cycle(len(models)-1, initial.copy(),
        MolecularResidual.zeros(initial), FASConfig(correction='component'),
        initial_residual=residual, return_residual=True)
    counts = [work_difference(m, b) for m, b in zip(models, before)]
    # Error diagnostics run after the observed cycle and are excluded from work.
    captures = [('initial', initial), *[(r['label'], r['state']) for r in hierarchy.phase_states]]
    rows = [dict(label=label, error_bands=error_bands(fine, state, reference)) for label, state in captures]
    write(directory/'result.json', dict(phases=rows, repair_records=hierarchy.repair_records,
        final_residual=defect.norm(fine.space), work_summary=work_summary(models, counts),
        scope='Base-cycle phase captures precede the optional horizontal correction'))
    save_state(directory/'selected.npz', final)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system', choices=(*SYSTEMS, 'He'), default='LiH')
    parser.add_argument('--level', type=int, choices=range(1, 6), default=3)
    parser.add_argument('--half-width', type=float, default=8.)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--geometry', choices=('canonical', 'rotated', 'distorted'), default='canonical')
    parser.add_argument('--start', choices=('continuation', 'continuation-perturbed', 'core', 'core-perturbed'), default='continuation')
    parser.add_argument('--continuation', choices=('fas', 'scf'))
    parser.add_argument('--reference-method', choices=('SCF', 'Newton-horizontal'), default='SCF')
    parser.add_argument('--methods', choices=METHODS, nargs='+', default=list(METHODS[:3]))
    parser.add_argument('--stage', choices=('all', 'prepare', 'solve', 'phases'), default='all')
    parser.add_argument('--inputs', type=Path, help='Optional existing initial/reference/ready/initialization directory')
    parser.add_argument('--base-inputs', type=Path, help='Prepare a prescribed start from canonical inputs, retaining the exact reference')
    parser.add_argument('--cache', type=Path, default=Path('results/fft-cache'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.workers < 1 or not np.isfinite(args.half_width) or args.half_width <= 0:
        parser.error('workers and half-width must be positive')
    if args.geometry != 'canonical' and args.system != 'CH4':
        parser.error('Geometry variants are defined for CH4 only')
    if len(args.methods) != len(set(args.methods)):
        parser.error('Methods must be distinct')
    if args.inputs and args.stage == 'prepare':
        parser.error('--inputs cannot be combined with --stage prepare')
    if args.base_inputs and (args.inputs or args.stage not in ('prepare', 'all')):
        parser.error('--base-inputs requires prepare/all and cannot be combined with --inputs')
    models, setup = build(args)
    args.output.mkdir(parents=True, exist_ok=True)
    gs_scan.warmup()
    origin = args.inputs or args.output/'inputs'
    if args.stage == 'prepare' or (args.stage == 'all' and args.inputs is None):
        prepare(args, models, origin)
    write(args.output/('run-'+args.stage+'.json'), dict(
        arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        setup=setup, host=platform.node(), job_id=os.getenv('SLURM_JOB_ID'),
        affinity=sorted(os.sched_getaffinity(0)), python=platform.python_version(),
        versions={p:importlib.metadata.version(p) for p in ('numpy', 'scipy')},
        gs_backend=gs_scan.backend(), threads={k:os.getenv(k) for k in
            ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
        source_sha256={str(p.relative_to(Path(__file__).parents[2])):digest(p)
            for p in sorted(Path(__file__).parents[1].rglob('*.py'))}))
    if args.stage == 'prepare':
        return
    initial, reference, ready = load_inputs(origin, models[-1])
    if args.stage == 'phases':
        phases(args, models, initial, reference)
    elif not solve(args, models, initial, reference, ready):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
