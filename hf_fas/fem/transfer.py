"""Independent orbital/pair mass-adjoint transfers on conforming mesh hierarchies."""
from dataclasses import dataclass
import numpy as np
from .model import MolecularState, MolecularResidual
from .records import timed


@dataclass
class FEMTransfer:
    coarse: object
    fine: object

    def __post_init__(self):
        self.orbital_prolongation = self.coarse.space.interpolation_to(self.fine.space, 1)
        self.pair_prolongation = self.coarse.space.interpolation_to(self.fine.space, 2)
        if self.coarse.molecule.occupations != self.fine.molecule.occupations or self.coarse.molecule.weights != self.fine.molecule.weights:
            raise ValueError('All hierarchy levels must have the same molecular spin occupations')

    def restrict_orbitals(self, values):
        return self.coarse.space.mass_inverse.solve(self.orbital_prolongation.T @ (self.fine.space.mass @ values))

    def restrict_pairs(self, values):
        shape = values.shape[1:]
        load = self.pair_prolongation.T @ (self.fine.space.pair_mass @ values.reshape(self.fine.space.n_pair, -1))
        return self.coarse.space.pair_mass_inverse.solve(load).reshape((self.coarse.space.n_pair,)+shape)

    def state(self, state):
        return MolecularState(tuple(self.restrict_orbitals(p) for p in state.orbitals),
                              tuple(self.restrict_pairs(w) for w in state.pairs),
                              tuple(l.copy() for l in state.multipliers))

    def residual(self, residual):
        return MolecularResidual(tuple(self.restrict_orbitals(p) for p in residual.orbitals),
                                 tuple(self.restrict_pairs(w) for w in residual.pairs),
                                 tuple(g.copy() for g in residual.grams))

    @timed('tau_transfer', lambda self, *args, **kwargs: self.fine.work)
    def coarse_equation(self, state, forcing, *, fine_residual=None):
        restricted = self.state(state)
        defect = forcing-self.fine.operator(state) if fine_residual is None else MolecularResidual.zeros(state)-fine_residual
        rhs = self.coarse.operator(restricted)+self.residual(defect)
        return restricted, rhs

    def prolong_physical(self, state):
        return self.fine.rebuild(tuple(self.orbital_prolongation @ p for p in state.orbitals))
