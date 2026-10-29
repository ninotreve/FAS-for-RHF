"""Real CLI solves and persisted input/endpoint contracts."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from hf_fas.experiments.reproduce import build, digest, load_inputs, read
from hf_fas.experiments.mechanism import spatial_error
from types import SimpleNamespace


@pytest.fixture(scope='module')
def completed(tmp_path_factory):
    folder = tmp_path_factory.mktemp('reproduction')/'He-L1'
    command = [sys.executable, '-m', 'hf_fas.experiments.reproduce', '--system', 'He',
               '--level', '1', '--half-width', '4', '--workers', '1',
               '--cache', str(folder.parent/'cache'), '--output', str(folder)]
    env = dict(os.environ, HF_FAS_GS_BACKEND='python', OPENBLAS_NUM_THREADS='1')
    subprocess.run(command, env=env, check=True, capture_output=True, text=True, timeout=180)
    return folder, command, env


def test_three_real_solvers_reproduce_energy_and_independent_endpoints(completed):
    folder, _, _ = completed
    rows = read(folder/'summary.json')
    assert [r['method'] for r in rows] == ['FAS-H', 'Newton-horizontal', 'SCF']
    # Fixed small-grid numerical baseline; deliberately do not compare wall time.
    expected_energy = -2.55849723334005
    ready = read(folder/'inputs/ready.json')
    assert read(folder/'inputs/initialization.json')['finest_solved'] is False
    for row in rows:
        assert row['qualified'] and row['converged']
        assert row['final_residual'] < 1e-8
        assert row['history'][-1]['electronic_energy'] == pytest.approx(expected_energy, abs=2e-9)
        assert row['endpoint']['qualified'] and row['precision_validation']['qualified']
        assert row['precision_validation']['vgf_precompute'] == 'fft'
        assert row['work_summary']['fft_work'] > 0
        assert row['application_seconds'] >= row['numerical_seconds'] > 0
        assert row['initial_sha256'] == ready['initial_sha256']
        assert row['reference_sha256'] == ready['reference_sha256']
        assert digest(folder/row['method']/'selected.npz') == row['selected_sha256']
    fas = rows[0]
    assert len(fas['repair_records']) == fas['iterations']


def test_cli_phase_capture_and_exact_reference_reuse(completed):
    folder, command, env = completed
    subprocess.run(command+['--stage', 'phases'], env=env, check=True,
                   capture_output=True, text=True, timeout=180)
    data = read(folder/'phases/result.json')
    labels = [r['label'] for r in data['phases']]
    assert labels[0:2] == ['initial', 'pre']
    assert any(label.endswith('-raw') for label in labels)
    for phase in data['phases']:
        b = phase['error_bands']
        assert b['total'] == pytest.approx(b['low']+b['middle']+b['high'], rel=3e-14, abs=1e-25)
    variant = folder.parent/'He-core'
    # Prepare another start without rerunning/changing the common reference.
    subprocess.run(command+['--output', str(variant), '--stage', 'prepare',
                           '--base-inputs', str(folder/'inputs'), '--start', 'core'],
                   env=env, check=True, capture_output=True, text=True, timeout=180)
    assert digest(variant/'inputs/reference.npz') == digest(folder/'inputs/reference.npz')
    assert read(variant/'inputs/initialization.json')['reference_used_to_construct_start'] is False


def test_corrupted_saved_start_is_rejected(completed, tmp_path):
    import shutil
    folder, _, _ = completed
    origin = tmp_path/'inputs'
    shutil.copytree(folder/'inputs', origin)
    models, _ = build(SimpleNamespace(system='He', level=1, half_width=4., workers=1,
                    geometry='canonical', cache=tmp_path/'cache'))
    with (origin/'initial.npz').open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='SHA256'):
        load_inputs(origin, models[-1])


def test_sine_bands_equal_independent_consistent_mass_norm():
    from hf_fas.tensor import TensorMesh, TensorQ1Space
    s = TensorQ1Space(TensorMesh(8, 4.))
    e = np.random.default_rng(138).normal(size=(s.n_orbital, 3))
    b = spatial_error(s, e)
    assert b['total'] == pytest.approx(np.sum(e*(s.mass@e)), rel=3e-14)
