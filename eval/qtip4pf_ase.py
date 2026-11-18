"""
Exact q-TIP4P/f water energies, forces and constant-energy dynamics from Python, for E22.

The water data were made by the i-PI simulation program talking to i-PI's Fortran force driver
(`i-pi-driver -m qtip4pf`) over a socket (data/water/run_one.sh). Here the same driver binary
is used, but the server side of the socket is ASE's SocketServer instead of i-PI, so energies
and forces come from exactly the same code that made the data, and short exact simulations can
be started from any state (for example a state predicted by a model) without writing i-PI
input files.

Dynamics: velocity Verlet with a 0.25 fs time step, as data/water/nve.xml.template. i-PI by
default keeps the centre-of-mass momentum fixed; ASE does not touch it. From a data frame the
centre-of-mass momentum is already zero, so the two agree (tested by check_against_ipi);
from a model's predicted state any centre-of-mass momentum is kept as the model left it.

Units: positions in Angstrom, momenta in ASE units (sqrt(eV amu)), energies in eV. Masses
must be passed explicitly (the config masses, H 1.00794 and O 15.9994, which are i-PI's).

Both checks below must pass before any E22 result is used:
  python eval/qtip4pf_ase.py /storage3/lauren/data/water/run_011 configs/flashmd_water16_scratch.yaml
"""

import argparse
import os
import subprocess
import sys
import uuid

import ase
import ase.units
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.socketio import SocketServer
from ase.md.verlet import VelocityVerlet

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from checks.check_water_run_io import P_AU_TO_ASE, read_ipi_xyz  # noqa: E402

DT_FS = 0.25


class QTIP4PF(Calculator):
    """ASE calculator backed by one i-pi-driver process (q-TIP4P/f). Use as a context manager."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, driver="i-pi-driver", log_path=None):
        super().__init__()
        # not "name": that is a read-only property of ASE calculators
        self.socket_name = f"e22_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        self.server = SocketServer(unixsocket=self.socket_name, timeout=120)
        self._log = open(log_path, "w") if log_path else subprocess.DEVNULL
        try:
            self.proc = subprocess.Popen([driver, "-u", "-a", self.socket_name, "-m", "qtip4pf"],
                                         stdout=self._log, stderr=subprocess.STDOUT)
        except BaseException:
            self.server.close()      # removes the socket file
            raise
        # lets ASE notice a driver that dies before connecting instead of waiting forever
        self.server.proc = self.proc

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if self.proc.poll() is not None:
            raise RuntimeError(f"i-pi-driver exited with code {self.proc.returncode}")
        r = self.server.calculate(self.atoms)
        self.results = {"energy": float(r["energy"]), "forces": np.array(r["forces"])}

    def close(self):
        self.server.proc = None      # ASE's close would wait on it; it is stopped below instead
        try:
            self.server.close()
        finally:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            if self._log is not subprocess.DEVNULL:
                self._log.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def make_atoms(numbers, positions, momenta, cell, masses):
    """masses: array per atom (amu). cell: (3,) box lengths or (3, 3)."""
    cell = np.diag(cell) if np.ndim(cell) == 1 else np.asarray(cell)
    a = ase.Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)
    a.set_masses(masses)
    if momenta is not None:
        a.set_momenta(momenta, apply_constraint=False)
    return a


def potential_energy(calc, numbers, positions, cell, masses):
    a = make_atoms(numbers, positions, None, cell, masses)
    a.calc = calc
    return a.get_potential_energy()


def exact_dynamics(calc, numbers, positions, momenta, cell, masses, n_steps, record_every=0, dt_fs=DT_FS):
    """Velocity Verlet for n_steps of dt_fs (0.25 fs, as the data) from (positions, momenta).
    Returns (q, p) at the end, and if record_every > 0 also lists of (q, p) every record_every
    steps including step 0."""
    a = make_atoms(numbers, positions, momenta, cell, masses)
    a.calc = calc
    md = VelocityVerlet(a, timestep=dt_fs * ase.units.fs)
    rec_q, rec_p = [], []
    if record_every:
        rec_q.append(a.get_positions().copy()); rec_p.append(a.get_momenta().copy())
        for _ in range(n_steps // record_every):
            md.run(record_every)
            rec_q.append(a.get_positions().copy()); rec_p.append(a.get_momenta().copy())
        rest = n_steps % record_every
        if rest:
            md.run(rest)
    else:
        md.run(n_steps)
    q, p = a.get_positions().copy(), a.get_momenta().copy()
    return (q, p, rec_q, rec_p) if record_every else (q, p)


def read_run(run_dir):
    """Raw i-PI output of one constant-energy run: positions (Angstrom), momenta (ASE units),
    atomic numbers, box lengths, and the potential energy column of nve.out (eV), one row per
    0.25 fs step."""
    from ase.data import atomic_numbers
    q, symbols, cells = read_ipi_xyz(os.path.join(run_dir, "nve.pos_0.xyz"))
    p, symbols_p, _ = read_ipi_xyz(os.path.join(run_dir, "nve.mom_0.xyz"))
    assert symbols == symbols_p and q.shape == p.shape
    assert np.allclose(cells, cells[0]), "box changes during the run"
    out = np.loadtxt(os.path.join(run_dir, "nve.out"), comments="#")
    numbers = np.array([atomic_numbers[s] for s in symbols])
    return {"q": q, "p": p * P_AU_TO_ASE, "numbers": numbers, "cell": cells[0], "potential": out[:, 5],
            "conserved": out[:, 2]}


def config_masses(config):
    import yaml
    m = yaml.safe_load(open(config))["architecture"]["training"]["masses"]
    return {int(k): float(v) for k, v in m.items()}


def stored_quantum(x):
    """i-PI writes xyz values as %12.5e (6 significant figures): the rounding step of each value."""
    a = np.maximum(np.abs(x), 1e-12)
    return 10.0 ** (np.floor(np.log10(a)) - 5)


def check_against_ipi(run_dir, config, n_energy=200, tol_energy=5e-3):
    """Checks that the energies and the dynamics here are the ones that made the data, with
    tolerances set by the precision of the stored frames (6 significant figures; positions are
    unwrapped, so most are stored to 1e-4 A and some to 1e-3 A):
    1. potential energy of n_energy stored frames equals i-PI's recorded value to tol_energy (the
       rounding of the stored positions alone gives about 1e-3 eV);
    2. 64 steps of exact dynamics from frame 0 reproduce the stored frame 64 (16 fs): root mean
       square position difference within 2 times the root mean square rounding step, and root mean
       square momentum difference within 1% of the momentum spread;
    3. must fail: the same with a 0.26 fs time step (about 1e-2 A off after 16 fs) fails check 2."""
    r = read_run(run_dir)
    mz = config_masses(config)
    masses = np.array([mz[int(z)] for z in r["numbers"]])
    assert len(r["potential"]) == len(r["q"]), (len(r["potential"]), len(r["q"]))
    with QTIP4PF() as calc:
        idx = np.linspace(0, len(r["q"]) - 1, n_energy).astype(int)
        e = np.array([potential_energy(calc, r["numbers"], r["q"][i], r["cell"], masses) for i in idx])
        de = np.abs(e - r["potential"][idx])
        q, p = exact_dynamics(calc, r["numbers"], r["q"][0], r["p"][0], r["cell"], masses, 64)
        qf, pf = exact_dynamics(calc, r["numbers"], r["q"][0], r["p"][0], r["cell"], masses, 64, dt_fs=0.26)
    tol_q = 2 * np.sqrt((stored_quantum(r["q"][64]) ** 2).mean())
    spread = np.sqrt((r["p"][64] ** 2).mean())
    rms = lambda d: float(np.sqrt((d ** 2).mean()))  # noqa: E731
    dq, dp = rms(q - r["q"][64]), rms(p - r["p"][64]) / spread
    dqf, dpf = rms(qf - r["q"][64]), rms(pf - r["p"][64]) / spread
    print(f"energy replay on {n_energy} frames: largest difference {de.max():.2e} eV (tolerance {tol_energy:.0e})")
    print(f"16 fs of exact dynamics from frame 0: position difference {dq:.2e} A (tolerance {tol_q:.1e}), "
          f"momentum difference {dp:.2e} of the momentum spread (tolerance 1e-2)")
    print(f"must fail, 0.26 fs time step: position difference {dqf:.2e} A, momentum difference {dpf:.2e}")
    passes = lambda a, b: a <= tol_q and b <= 1e-2  # noqa: E731
    ok = de.max() <= tol_energy and passes(dq, dp) and not passes(dqf, dpf)
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("config")
    args = ap.parse_args()
    sys.exit(0 if check_against_ipi(args.run_dir, args.config) else 1)
