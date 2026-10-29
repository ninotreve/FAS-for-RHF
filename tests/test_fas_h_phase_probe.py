"""Instrumentation must preserve production FAS-H states and work."""
import numpy as np
import pytest

from hf_fas.fem import FASConfig, MolecularFEMModel, MolecularResidual, regression_molecules
from hf_fas.fem.records import model_work, work_difference
from hf_fas.tensor import TensorCoulomb, TensorMesh, TensorQ1Space
from hf_fas.experiments.fem_run import state_arrays
from hf_fas.experiments.q1_boundary_repairs import POLICY, RepairHierarchy
from hf_fas.experiments.q1_fas_h_phase_probe import ObservedRepairHierarchy


def counts(work):
    return [{group: {key: value for key, value in entries.items() if not key.startswith('time.')}
             for group, entries in level.items()} for level in work]


@pytest.mark.parametrize('variant,trigger', [('C', 0.), ('C-response', 0.), ('C-response', 1e9)])
def test_observed_cycle_preserves_production_and_counters(variant, trigger):
    models = []
    for n in (4, 8):
        space = TensorQ1Space(TensorMesh(n, 4.))
        models.append(MolecularFEMModel(space, regression_molecules()['He'], TensorCoulomb(space)))
    initial = models[-1].initial_state()
    forcing = MolecularResidual.zeros(initial)
    cfg = FASConfig(correction='component')

    def run(cls):
        hierarchy = cls(models, variant, dict(POLICY, trigger_ratio=trigger))
        before = [model_work(m) for m in models]
        defect = models[-1].residual(initial)
        state, residual = hierarchy.cycle(1, initial.copy(), forcing, cfg,
                                          initial_residual=defect, return_residual=True)
        return hierarchy, state, residual, counts([work_difference(m, b) for m, b in zip(models, before)])

    ordinary, expected, defect, work = run(RepairHierarchy)
    observed, actual, observed_defect, observed_work = run(ObservedRepairHierarchy)
    for key, value in state_arrays(expected).items():
        np.testing.assert_array_equal(value, state_arrays(actual)[key])
    assert defect.norm(models[-1].space) == observed_defect.norm(models[-1].space)
    assert work == observed_work
    assert ordinary.cycle_records == observed.cycle_records
    assert ordinary.smoother_records == observed.smoother_records
    if variant == 'C':
        assert not ordinary.repair_records and not observed.repair_records
    else:
        assert ordinary.repair_records[0]['triggered'] == observed.repair_records[0]['triggered']
    assert observed.fas_work is not None
    assert observed.phase_states[0]['label'] == 'pre'
    assert any(r['label'].endswith('-raw') for r in observed.phase_states)
    assert any(r['label'].startswith('post_') and not r['label'].endswith('-raw') for r in observed.phase_states)
    saved = observed.phase_states[-1]['state'].orbitals[0].copy()
    actual.orbitals[0][:] = 0
    np.testing.assert_array_equal(observed.phase_states[-1]['state'].orbitals[0], saved)
