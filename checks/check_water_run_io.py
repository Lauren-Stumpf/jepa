"""Shared reader and unit constant for i-PI water output (used by checks and visualisation)."""
import numpy as np
import ase.units as U

BOHR_A = 0.529177210903
AMU_ME = 1822.888486209
AU_TIME_FS = 0.0241888432650516
# i-PI momentum atomic unit -> ASE internal momentum unit (sqrt(eV*amu))
P_AU_TO_ASE = (BOHR_A / AU_TIME_FS) * (1.0 / AMU_ME) * (1.0 / U.fs)


def read_ipi_xyz(path):
    """Return (frames[nframes, natoms, 3], symbols, cell_lengths[nframes, 3])."""
    frames, cells = [], []
    symbols = None
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            n = int(line)
            toks = f.readline().split()
            i = toks.index("CELL(abcABC):")
            cells.append([float(toks[i + 1]), float(toks[i + 2]), float(toks[i + 3])])
            syms, xyz = [], []
            for _ in range(n):
                t = f.readline().split()
                syms.append(t[0])
                xyz.append([float(t[1]), float(t[2]), float(t[3])])
            if symbols is None:
                symbols = syms
            frames.append(xyz)
    return np.array(frames), symbols, np.array(cells)
