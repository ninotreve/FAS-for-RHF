"""Paper FAS-H correction, preserved from the C-response implementation."""
from dataclasses import asdict
import numpy as np
from hf_fas.fem import FEMHierarchy, solve_molecular_quotient
from hf_fas.fem.coarse import CoarseConfig

POLICY = dict(trigger_ratio=.4, krylov_iterations=60, krylov_restart=60,
              forcing_ceiling=.01, krylov_tolerance=1e-8)

def physical(forcing):
    return not any(np.any(x) for g in (forcing.orbitals,forcing.pairs,forcing.grams) for x in g)

class RepairHierarchy(FEMHierarchy):
    """Extra finest post sweeps or one bounded physical horizontal correction.

    Joint forced coarse equations and the default hierarchy are unchanged.
    A response correction is attempted only after a complete physical finest
    cycle leaves at least trigger_ratio of its entering residual.
    """
    def __init__(self,models,variant='C',policy=None):
        super().__init__(models);self.variant=variant;self.policy=dict(POLICY if policy is None else policy)
        if variant not in ('C','C-post4','C-response'):raise ValueError('Unknown repair policy')
        self.repair_records=[]

    def _relax(self,level,state,forcing,sweeps,cfg,phase,residual=None):
        if self.variant=='C-post4' and level==len(self.models)-1 and phase!='pre':sweeps+=2
        return super()._relax(level,state,forcing,sweeps,cfg,phase,residual)

    def cycle(self,level,state,forcing,cfg,*,initial_residual=None,return_residual=False):
        fine=level==len(self.models)-1
        if fine and self.variant=='C-response' and not physical(forcing):
            raise ValueError('Physical response repair cannot be used on a forced root')
        baseline=(initial_residual if initial_residual is not None else
                  self.models[level].residual(state,forcing)).norm(self.models[level].space) if fine else None
        output,defect=super().cycle(level,state,forcing,cfg,
            initial_residual=initial_residual,return_residual=True)
        if fine and self.variant=='C-response':
            merit=defect.norm(self.models[level].space)
            report=dict(cycle=self.current_cycle,cycle_entry_residual=baseline,
                FAS_residual=merit,triggered=bool(merit>=self.policy['trigger_ratio']*baseline and merit>cfg.tolerance),
                policy=self.policy)
            if report['triggered']:
                ncfg=CoarseConfig(tolerance=cfg.tolerance,max_iterations=1,gs_fallback=False,**{
                    k:self.policy[k] for k in ('krylov_iterations','krylov_restart','forcing_ceiling','krylov_tolerance')})
                result=solve_molecular_quotient(self.models[level],output,ncfg)
                candidate=self.models[level].residual(result.state)
                value=candidate.norm(self.models[level].space)
                report.update(history=result.history,seconds=result.elapsed_seconds,
                    accepted=bool(value<merit),candidate_residual=value,Newton_config=asdict(ncfg))
                if value<merit:output,defect=result.state,candidate
            report['final_residual']=defect.norm(self.models[level].space)
            self.repair_records.append(report)
        return (output,defect) if return_residual else output
