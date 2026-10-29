"""Atomic JSON and state persistence for reproduction runs."""
import os
import json
import tempfile
from pathlib import Path
import numpy as np

def write(path, value):
    def plain(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, dict): return {k:plain(v) for k,v in x.items()}
        if isinstance(x, (tuple, list)): return [plain(v) for v in x]
        return x
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(plain(value), indent=2, allow_nan=False)+'\n'); temp.replace(path)

def state_arrays(state):
    return {f'{prefix}_{i}':value for prefix,group in
            (('phi',state.orbitals),('pair',state.pairs),('lambda',state.multipliers))
            for i,value in enumerate(group)}

def save_state(path, state):
    """Readers see the old or new complete archive, never a partial checkpoint."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.state-',suffix='.npz',delete=False) as handle:
            temporary = Path(handle.name)
            np.savez(handle, **state_arrays(state))
        os.replace(temporary,path)
    finally:
        if temporary is not None and temporary.exists(): temporary.unlink()
