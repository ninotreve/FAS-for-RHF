"""Observe one production FAS-H cycle without evaluating operators in it."""
from hf_fas.fem import FEMHierarchy
from hf_fas.fem.records import model_work
from .q1_boundary_repairs import RepairHierarchy


class PhaseCaptureBase(FEMHierarchy):
    """Record finest-grid state copies and counters around existing stages."""
    def __init__(self, models):
        super().__init__(models)
        self.phase_states = []
        self.fas_work = None

    def _capture(self, label, state, residual):
        self.phase_states.append(dict(label=label, state=state.copy(),
            residual=None if residual is None else residual.copy(),
            counters=[model_work(m) for m in self.models]))

    def _relax(self, level, state, forcing, sweeps, cfg, phase, residual=None):
        finest = level == len(self.models) - 1
        if finest and phase != 'pre':
            self._capture(phase + '-raw', state, residual)
        output, defect = super()._relax(level, state, forcing, sweeps, cfg, phase, residual)
        if finest:
            self._capture(phase, output, defect)
        return output, defect

    def cycle(self, level, state, forcing, cfg, **kwargs):
        output, defect = super().cycle(level, state, forcing, cfg, **dict(kwargs, return_residual=True))
        if level == len(self.models) - 1:
            self.fas_work = [model_work(m) for m in self.models]
        return (output, defect) if kwargs.get('return_residual') else output


class ObservedRepairHierarchy(RepairHierarchy, PhaseCaptureBase):
    """Observe the FAS stages before RepairHierarchy's physical correction."""
