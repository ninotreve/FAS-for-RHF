"""Uniform conforming hexahedra, with eliminated orbital boundary nodes."""
from dataclasses import dataclass
import hashlib
import numpy as np


@dataclass(frozen=True)
class TensorMesh:
    elements: int = 4
    half_width: float = 8.

    def __post_init__(self):
        if not isinstance(self.elements, int) or self.elements < 2 or not np.isfinite(self.half_width) or self.half_width <= 0:
            raise ValueError('Positive box width and at least two elements per axis required')

    @property
    def spacing(self):
        return 2*self.half_width/self.elements

    @property
    def axis(self):
        return np.linspace(-self.half_width, self.half_width, self.elements+1)

    @property
    def fingerprint(self):
        return hashlib.sha256(repr(('uniform_Q1_hex_v1', self.elements, self.half_width)).encode()).hexdigest()

    def uniform_refine(self):
        return TensorMesh(2*self.elements, self.half_width)

    def quality(self):
        return dict(cells=self.elements**3, h_max=float(np.sqrt(3)*self.spacing),
                    h_min=float(np.sqrt(3)*self.spacing), minimum_shape_ratio=float(1/np.sqrt(3)))
