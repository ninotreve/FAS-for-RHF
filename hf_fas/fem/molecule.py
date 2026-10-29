"""Molecular charge/spin bookkeeping independent of named benchmarks."""
from dataclasses import dataclass
import numpy as np

SYMBOLS = ('X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og').split()


@dataclass(frozen=True)
class Molecule:
    atomic_numbers: np.ndarray
    centers: np.ndarray
    charge: int = 0
    spin: int = 0
    restricted: bool | None = None

    def __post_init__(self):
        numbers = np.asarray(self.atomic_numbers)
        centers = np.asarray(self.centers, dtype=float)
        if numbers.ndim != 1 or len(numbers) == 0 or centers.shape != (len(numbers), 3):
            raise ValueError('One three-dimensional position is required per nucleus')
        if not np.isfinite(centers).all() or np.any(numbers != numbers.astype(int)) or np.any((numbers < 1) | (numbers > 118)):
            raise ValueError('Invalid nuclear positions or atomic numbers')
        if int(self.charge) != self.charge or int(self.spin) != self.spin:
            raise ValueError('Charge and spin must be integers')
        electrons = int(numbers.sum())-int(self.charge)
        if electrons < 0 or abs(self.spin) > electrons or (electrons+self.spin) % 2:
            raise ValueError('Inconsistent electron count and N_alpha-N_beta spin')
        restricted = (self.spin == 0 and electrons % 2 == 0) if self.restricted is None else bool(self.restricted)
        if restricted and (self.spin != 0 or electrons % 2):
            raise ValueError('Restricted closed-shell HF requires even electrons and spin zero; use unrestricted')
        if len(centers) > 1:
            radius = np.linalg.norm(centers[:, None]-centers[None], axis=-1)
            np.fill_diagonal(radius, np.inf)
            if np.min(radius) == 0:
                raise ValueError('Coincident nuclei')
        numbers = numbers.astype(int).copy(); centers = centers.copy()
        numbers.flags.writeable = centers.flags.writeable = False
        object.__setattr__(self, 'atomic_numbers', numbers)
        object.__setattr__(self, 'centers', centers)
        object.__setattr__(self, 'restricted', restricted)

    @property
    def electrons(self):
        return int(self.atomic_numbers.sum())-self.charge

    @property
    def occupations(self):
        return (self.electrons//2,) if self.restricted else ((self.electrons+self.spin)//2, (self.electrons-self.spin)//2)

    @property
    def weights(self):
        return (2.,) if self.restricted else (1., 1.)

    @property
    def nuclear_repulsion(self):
        return float(sum(self.atomic_numbers[i]*self.atomic_numbers[j]/np.linalg.norm(self.centers[i]-self.centers[j]) for i in range(len(self.centers)) for j in range(i)))

    @classmethod
    def from_atoms(cls, atoms, *, unit='bohr', **kwargs):
        numbers, centers = [], []
        for symbol, position in atoms:
            number = SYMBOLS.index(symbol) if isinstance(symbol, str) else int(symbol)
            numbers.append(number); centers.append(position)
        if unit not in ('bohr', 'angstrom'):
            raise ValueError('Coordinate unit must be bohr or angstrom')
        # Fixed atomic-unit conversion, not a mesh-dependent rescaling.
        coordinates = np.asarray(centers, dtype=float)*(1.8897261254578281 if unit == 'angstrom' else 1.)
        return cls(np.asarray(numbers), coordinates, **kwargs)


def regression_molecules():
    angle = np.deg2rad(104.5/2); bond = 1.8088458466
    return dict(He=Molecule.from_atoms([('He', (0, 0, 0))]),
                LiH=Molecule.from_atoms([('Li', (-1.5075, 0, 0)), ('H', (1.5075, 0, 0))]),
                BH=Molecule.from_atoms([('B', (-1.168, 0, 0)), ('H', (1.168, 0, 0))]),
                H2O=Molecule.from_atoms([('O', (0, 0, 0)), ('H', (bond*np.sin(angle), 0, bond*np.cos(angle))), ('H', (-bond*np.sin(angle), 0, bond*np.cos(angle)))]))


def paper_molecules():
    """Fixed experimental panel; geometries in bohr, not optimized geometries.

    Preserve the four-system regression defaults. BeH+ is a four-electron
    isoelectronic control for LiH; Li2 and BeH2 have six electrons. A shared
    solver policy is used across all systems, including unsuccessful cases.
    """
    return dict(regression_molecules(),
        BeHplus=Molecule.from_atoms([('Be', (-1.25, 0, 0)), ('H', (1.25, 0, 0))], charge=1),
        Li2=Molecule.from_atoms([('Li', (-2.525, 0, 0)), ('Li', (2.525, 0, 0))]),
        BeH2=Molecule.from_atoms([('Be', (0, 0, 0)), ('H', (-2.54, 0, 0)), ('H', (2.54, 0, 0))]),
        C2=Molecule.from_atoms([('C', (-1.175, 0, 0)), ('C', (1.175, 0, 0))]),
        N2=Molecule.from_atoms([('N', (-1.05, 0, 0)), ('N', (1.05, 0, 0))]))
