"""Reproducible coarse-to-fine initialization before a shared finest start."""
from dataclasses import dataclass, field
from time import perf_counter
import numpy as np
from .fas import FEMHierarchy, FASConfig, solve_molecular_fas
from .scf import SCFConfig, solve_molecular_scf, occupied_eigensolve
from .geometry import projector_distance, InvalidTrialState


@dataclass(frozen=True)
class InitializationConfig:
    seed: int = 20261014
    candidate_amplitudes: tuple = (0.,.05)
    scf: SCFConfig = field(default_factory=lambda:SCFConfig(tolerance=1e-9))
    fas: FASConfig = field(default_factory=lambda:FASConfig(tolerance=1e-9,max_cycles=100))
    continuation_method: str = 'fas'


def initialize_hierarchy(hierarchy, config=None, callback=None):
    """Solve lower levels, then only prolong to the comparison level.

    Intermediate nonlinear solves use the explicit FAS or SCF policy.
    No finest SCF endpoint enters this start. Candidate and continuation
    failures are returned explicitly, and initialization time is charged once.
    """
    cfg = config or InitializationConfig(); started = perf_counter()
    if cfg.continuation_method not in ('fas','scf'):
        raise ValueError('Initialization continuation must be fas or scf')
    provenance = dict(continuation_method=cfg.continuation_method,finest_solved=False)
    if len(hierarchy.models) < 2:
        raise ValueError('Common-start initialization requires at least two levels')
    coarse = hierarchy.models[0]; base = coarse.initial_state()
    candidates,states = [],[]
    for index,amplitude in enumerate(cfg.candidate_amplitudes):
        rng = np.random.default_rng(cfg.seed+index)
        frames = []
        for p in base.orbitals:
            noise = rng.normal(size=p.shape)
            norm = np.sqrt(max(0.,np.sum(noise*(coarse.space.mass @ noise))))
            frames.append(p+amplitude*np.sqrt(p.shape[1])*noise/max(norm,1e-30))
        try:
            initial = coarse.rebuild(tuple(frames))
            def coarse_observer(state,record):
                if callback: callback(state,dict(record,phase=f'coarse_candidate_{index}',level=0))
            result = solve_molecular_scf(coarse,initial_state=initial,config=cfg.scf,callback=coarse_observer)
            record = dict(candidate=index,amplitude=amplitude,converged=result.converged,
                residual=result.final_residual,electronic_energy=coarse.energy(result.state).electronic,
                seconds=result.elapsed_seconds,work=result.work,history=result.history)
            candidates.append(record)
            if result.converged: states.append((record,result.state))
        except InvalidTrialState as exc:
            candidates.append(dict(candidate=index,amplitude=amplitude,converged=False,reason=str(exc)))
    if not states:
        return None,dict(provenance,converged=False,stage='coarse_candidates',candidates=candidates,
                         seconds=perf_counter()-started)
    selected,state = min(states,key=lambda item:(item[0]['electronic_energy'],item[0]['residual'],item[0]['candidate']))
    for record,other in states:
        record['distance_to_selected'] = max(projector_distance(coarse.space,a,b)
            for a,b in zip(state.orbitals,other.orbitals))
    continuation = []
    for level,transfer in enumerate(hierarchy.transfers,1):
        state = transfer.prolong_physical(state)
        model=hierarchy.models[level]; frames=[]; inner=[]
        for spin,p in enumerate(state.orbitals):
            action=lambda x,spin=spin:model.fock_load(state,spin,x)
            vectors,record=occupied_eigensolve(model,action,p,cfg.scf,1e-8,model.space.kinetic_inverse)
            frames.append(vectors); inner.append(record)
        state=model.rebuild(tuple(frames))
        selection=dict(level=level,frozen_Fock_selection=inner)
        continuation.append(selection)
        if not all(record['converged'] for record in inner):
            return None,dict(provenance,converged=False,stage='frozen_Fock_selection',level=level,candidates=candidates,
                             continuation=continuation,seconds=perf_counter()-started)
        if level == len(hierarchy.models)-1:
            break
        def continuation_observer(state,record):
            if callback: callback(state,dict(record,phase=f'continuation_{level}',level=level,
                                            continuation_method=cfg.continuation_method))
        if cfg.continuation_method == 'scf':
            result = solve_molecular_scf(model,initial_state=state,config=cfg.scf,callback=continuation_observer)
        else:
            local = FEMHierarchy(hierarchy.models[:level+1])
            result = solve_molecular_fas(local,initial_state=state,config=cfg.fas,callback=continuation_observer)
        selection.update(method=cfg.continuation_method,converged=result.converged,residual=result.final_residual,
            seconds=result.elapsed_seconds,work=result.work,history=result.history)
        if not result.converged:
            return None,dict(provenance,converged=False,stage='continuation',level=level,candidates=candidates,
                             continuation=continuation,seconds=perf_counter()-started)
        state = result.state
    return state,dict(provenance,converged=True,selected_candidate=selected['candidate'],candidates=candidates,
        continuation=continuation,seconds=perf_counter()-started,
        selection_scope='Lowest energy among converged prescribed coarse candidates; no global-ground-state claim')
