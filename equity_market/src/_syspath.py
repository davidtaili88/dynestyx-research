"""Path shim so the modules under src/ can keep FLAT imports (``from data
import ...``, ``from pagan_sossounov import ...``, ``from regime_model_2state
import fit``) even though they now live in sibling SUBFOLDERS of src/.

WHY THIS EXISTS. Running ``python analysis/stationarity_vol.py`` puts only
``src/analysis/`` on sys.path -- not ``src/`` and not the sibling folders -- so a
bare ``import data`` (which resolves to ``src/dataset/data.py``) would fail with
ModuleNotFoundError. Rather than convert src/ into a package and rewrite every
import to a qualified form (which would also force ``python -m ...`` invocation
everywhere), we simply add each subfolder back onto sys.path here. Every script
does ``import _syspath`` as its first project import; the two-line header above
that import makes ``src/`` itself importable so this file can be found.

To add a new subfolder of modules, just create it -- it is picked up
automatically (every immediate subdirectory of src/ is added).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent

for _child in _SRC.iterdir():
    if _child.is_dir() and not _child.name.startswith(("_", ".")):
        _p = str(_child)
        if _p not in sys.path:
            sys.path.insert(0, _p)
