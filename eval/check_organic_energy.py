"""
Is each organic run deterministic and energy-conserving? (E10, hypothesis H3)

Uses the catalogue of the source FlashMD file (results/data/catalogue_8fs.csv: kinetic
temperature from the stored momenta and the stored potential energy per atom) for every
frame of the organic subset (data/organic8/frames.csv, written frames of all splits).
Total energy per atom = 1.5 k_B T + potential energy per atom.

The source file holds, per molecule, 500 frames = 10 runs of 50 frames (the block structure
is checked here, not assumed: the energy must be constant inside a block and jump between
blocks). For each 50-frame block it reports the spread of the total energy (largest minus
smallest) and the drift (last minus first). A run with a thermostat, random kicks or rough
forces shows a spread far above the size of rounding in the stored numbers.

The other half of the check, that the stored 8 fs futures are reproduced exactly by rerunning
PET-MAD from the stored starts, was run on 30 Oct (results/eval/verlet_8fs.csv, 200 frames,
eval/eval_onestep.py --model verlet) and is summarised here too.

  python eval/check_organic_energy.py
"""

import json
import os

import ase.units
import numpy as np
import pandas as pd

RUN_FRAMES = 50


def main():
    cat = pd.read_csv("results/data/catalogue_8fs.csv", usecols=["frame", "group", "natoms", "kinetic_T_K", "pe_eV_atom"])
    fr = pd.read_csv("data/organic8/frames.csv")
    keep = fr[fr.written == 1].frame.to_numpy()
    d = cat.set_index("frame").loc[keep].reset_index()
    d["e_tot"] = 1.5 * ase.units.kB * d.kinetic_T_K + d.pe_eV_atom
    d["block"] = d.frame // RUN_FRAMES
    blocks = d.groupby("block").agg(n=("frame", "size"), group=("group", "first"), groups=("group", "nunique"),
                                    spread=("e_tot", lambda e: e.max() - e.min()),
                                    drift=("e_tot", lambda e: e.iloc[-1] - e.iloc[0]),
                                    T=("kinetic_T_K", "mean"))
    assert (blocks.groups == 1).all(), "a 50-frame block spans two structures"
    # jumps between consecutive blocks of the same structure: should be large if blocks are separate runs
    jumps = []
    for g, b in d.groupby("group"):
        means = b.groupby("block").e_tot.mean().to_numpy()
        jumps += list(np.abs(np.diff(means)))
    q = lambda x, p: float(np.quantile(x, p))  # noqa: E731
    res = {
        "blocks": int(len(blocks)), "frames": int(len(d)), "blocks_not_50_frames": int((blocks.n != RUN_FRAMES).sum()),
        "spread_eV_per_atom": {"median": q(blocks.spread, 0.5), "p99": q(blocks.spread, 0.99), "max": float(blocks.spread.max())},
        "drift_eV_per_atom": {"median_abs": q(blocks.drift.abs(), 0.5), "max_abs": float(blocks.drift.abs().max())},
        "jump_between_blocks_eV_per_atom": {"median": q(jumps, 0.5), "min": float(np.min(jumps))} if jumps else {},
        "worst_blocks": blocks.sort_values("spread", ascending=False).head(5).reset_index().to_dict("records"),
    }
    v = "results/eval/verlet_8fs.csv"
    if os.path.exists(v):
        r = pd.read_csv(v)
        res["rerun_200_frames"] = {"rmse_dq_massscaled_median": float(r.rmse_dq_massscaled.median()),
                                   "rmse_dq_massscaled_max": float(r.rmse_dq_massscaled.max()),
                                   "rmse_dp_massscaled_median": float(r.rmse_dp_massscaled.median()),
                                   "rmse_dp_massscaled_max": float(r.rmse_dp_massscaled.max())}
    print(json.dumps(res, indent=1, default=float))
    os.makedirs("results/eval", exist_ok=True)
    json.dump(res, open("results/eval/organic_energy_check.json", "w"), indent=1, default=float)
    print("written results/eval/organic_energy_check.json")


if __name__ == "__main__":
    main()
