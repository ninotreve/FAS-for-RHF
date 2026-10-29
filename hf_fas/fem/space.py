"""Shared discretization configuration for the Q1 tensor space."""
from dataclasses import dataclass

@dataclass(frozen=True)
class FEMConfig:
    mass_method: str = 'direct'
    auxiliary_method: str = 'amg'
    linear_tolerance: float = 1e-12
    pair_shift: float = 1.
    nuclear_order: int = 10
