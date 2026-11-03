"""
Pictures of one q-TIP4P/f water run, to understand what the model will learn.

Produces, in the run directory:
  vis_box_3d.png        the 64-water box in 3D, atoms coloured by speed
  vis_stride_arrows.png one frame with an arrow per atom to its position one
                        stride later (default 8 fs), raw and mass-scaled
  vis_traces.png        x-coordinate of one oxygen and its two hydrogens over
                        200 fs, with 1, 4, 8, 16 fs strides marked
  vis_distributions.png histograms of per-atom displacement and momentum over
                        the stride, by element, raw and mass-scaled
  vis_vacf.png          hydrogen and oxygen velocity autocorrelation and its
                        spectrum (shows the ~10 fs O-H stretch period)
  vis_water.html        interactive 3D box (plotly), rotatable in a browser

Usage:
  python eval/visualise_water_run.py /storage3/lauren/data/water/run_001 --stride_fs 8
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ase.units as U

import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_water_run_io import read_ipi_xyz, P_AU_TO_ASE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--stride_fs", type=float, default=8.0)
    ap.add_argument("--dt_fs", type=float, default=0.25)
    ap.add_argument("--frame", type=int, default=4000)
    args = ap.parse_args()
    d = args.run_dir
    q, syms, cells = read_ipi_xyz(os.path.join(d, "nve.pos_0.xyz"))
    p = read_ipi_xyz(os.path.join(d, "nve.mom_0.xyz"))[0] * P_AU_TO_ASE
    L = cells[0]
    syms = np.array(syms)
    masses = np.where(syms == "O", 15.9994, 1.00794)
    k = args.frame
    s = int(round(args.stride_fs / args.dt_fs))
    is_o = syms == "O"

    # wrap the chosen frame into the box by molecule (keep O H H together)
    fr = q[k].copy()
    for o in np.where(is_o)[0]:
        shift = L * np.floor(fr[o] / L)
        fr[o:o + 3] -= shift
    speed = np.linalg.norm(p[k] / masses[:, None], axis=1)

    # 1. 3D box coloured by speed
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(fr[is_o, 0], fr[is_o, 1], fr[is_o, 2], s=90, c=speed[is_o], cmap="viridis",
                    vmin=speed.min(), vmax=speed.max(), edgecolor="k", linewidth=0.3, label="O")
    ax.scatter(fr[~is_o, 0], fr[~is_o, 1], fr[~is_o, 2], s=25, c=speed[~is_o], cmap="viridis",
               vmin=speed.min(), vmax=speed.max(), edgecolor="k", linewidth=0.2, label="H")
    for o in np.where(is_o)[0]:
        for h in (o + 1, o + 2):
            ax.plot(*zip(fr[o], fr[h]), c="grey", lw=0.6)
    for a in (0, L[0]):
        for b in (0, L[1]):
            ax.plot([a, a], [b, b], [0, L[2]], c="k", lw=0.4)
            ax.plot([a, a], [0, L[1]], [b, b], c="k", lw=0.4)
            ax.plot([0, L[0]], [a, a], [b, b], c="k", lw=0.4)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="speed (ASE units)")
    ax.set_title(f"64 water molecules, frame {k} ({k*args.dt_fs:.0f} fs); large = O, small = H")
    ax.set_xlabel("x (A)"); ax.set_ylabel("y (A)"); ax.set_zlabel("z (A)")
    fig.tight_layout(); fig.savefig(os.path.join(d, "vis_box_3d.png"), dpi=130); plt.close(fig)

    # 2. stride arrows, raw and mass-scaled, projected on xy
    dq = q[k + s] - q[k]
    dq_ms = dq * np.sqrt(masses)[:, None]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, vec, title in ((axes[0], dq, f"displacement over {args.stride_fs:.0f} fs (A)"),
                           (axes[1], dq_ms, "mass-scaled displacement (A sqrt(amu))")):
        ax.scatter(fr[is_o, 0], fr[is_o, 1], s=60, c="tab:red", label="O", zorder=3)
        ax.scatter(fr[~is_o, 0], fr[~is_o, 1], s=15, c="tab:blue", label="H", zorder=3)
        ax.quiver(fr[:, 0], fr[:, 1], vec[:, 0], vec[:, 1], angles="xy", scale_units="xy", scale=0.15,
                  width=0.003, color="k")
        ax.set_aspect("equal"); ax.set_xlim(0, L[0]); ax.set_ylim(0, L[1])
        ax.set_title(title + "  (arrows x6.7)"); ax.legend(loc="upper right")
    fig.suptitle("one training pair: where each atom is now, and where it is one stride later")
    fig.tight_layout(); fig.savefig(os.path.join(d, "vis_stride_arrows.png"), dpi=130); plt.close(fig)

    # 3. traces over 200 fs for one molecule
    n200 = int(200 / args.dt_fs)
    t = np.arange(n200) * args.dt_fs
    o = np.where(is_o)[0][0]
    fig, ax = plt.subplots(figsize=(11, 4))
    for idx, lab, c in ((o, "O", "tab:red"), (o + 1, "H1", "tab:blue"), (o + 2, "H2", "tab:cyan")):
        ax.plot(t, q[k:k + n200, idx, 0] - q[k, idx, 0], label=lab, c=c, lw=1)
    for stride in (1, 4, 8, 16):
        ax.axvline(stride, c="grey", ls="--", lw=0.6)
        ax.text(stride, ax.get_ylim()[1] * 0.95, f"{stride} fs", fontsize=8, ha="left")
    ax.set_xlabel("time (fs)"); ax.set_ylabel("x displacement from start (A)")
    ax.set_title("one water molecule over 200 fs: hydrogens oscillate every ~10 fs, oxygen moves slowly")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(d, "vis_traces.png"), dpi=130); plt.close(fig)

    # 4. distributions over the stride, sampled every 100 fs
    idx = np.arange(0, len(q) - s, int(100 / args.dt_fs))
    dq_all = q[idx + s] - q[idx]
    p_all = p[idx + s]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for row, (arr, name) in enumerate(((dq_all, "displacement"), (p_all, "momentum"))):
        mag = np.linalg.norm(arr, axis=-1)
        ms = mag * np.sqrt(masses)[None, :] if name == "displacement" else mag / np.sqrt(masses)[None, :]
        for col, (vals, lab) in enumerate(((mag, f"raw {name} magnitude"), (ms, f"mass-scaled {name} magnitude"))):
            ax = axes[row, col]
            ax.hist(vals[:, is_o].ravel(), bins=60, alpha=0.6, label="O", density=True)
            ax.hist(vals[:, ~is_o].ravel(), bins=60, alpha=0.6, label="H", density=True)
            ax.set_xlabel(lab); ax.legend()
    fig.suptitle(f"what the loss sees: per-atom targets over an {args.stride_fs:.0f} fs stride; "
                 "mass scaling puts O and H on a similar footing")
    fig.tight_layout(); fig.savefig(os.path.join(d, "vis_distributions.png"), dpi=130); plt.close(fig)

    # 5. velocity autocorrelation and spectrum
    v = p / masses[None, :, None]
    nlag = int(100 / args.dt_fs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for mask, lab in ((is_o, "O"), (~is_o, "H")):
        vv = v[:, mask, :]
        c = np.array([(vv[:len(vv) - lag] * vv[lag:]).sum(-1).mean() for lag in range(nlag)])
        c /= c[0]
        axes[0].plot(np.arange(nlag) * args.dt_fs, c, label=lab)
        spec = np.abs(np.fft.rfft(c * np.hanning(nlag)))
        freq = np.fft.rfftfreq(nlag, args.dt_fs * 1e-15) / 2.99792458e10  # cm^-1
        axes[1].plot(freq, spec / spec.max(), label=lab)
    axes[0].set_xlabel("lag (fs)"); axes[0].set_ylabel("velocity autocorrelation"); axes[0].legend()
    axes[1].set_xlabel("wavenumber (cm^-1)"); axes[1].set_xlim(0, 4500); axes[1].set_ylabel("spectrum (normalised)")
    axes[1].axvline(3600, c="grey", ls="--", lw=0.6); axes[1].text(3600, 0.9, "O-H stretch", fontsize=8)
    axes[1].legend()
    fig.suptitle("how fast the motion decorrelates: this is the dynamical quantity the models will be tested on")
    fig.tight_layout(); fig.savefig(os.path.join(d, "vis_vacf.png"), dpi=130); plt.close(fig)

    # 6. interactive 3D
    try:
        import plotly.graph_objects as go
        traces = [go.Scatter3d(x=fr[is_o, 0], y=fr[is_o, 1], z=fr[is_o, 2], mode="markers",
                               marker=dict(size=7, color="red"), name="O"),
                  go.Scatter3d(x=fr[~is_o, 0], y=fr[~is_o, 1], z=fr[~is_o, 2], mode="markers",
                               marker=dict(size=3, color="lightblue"), name="H")]
        bx, by, bz = [], [], []
        for o in np.where(is_o)[0]:
            for h in (o + 1, o + 2):
                bx += [fr[o, 0], fr[h, 0], None]; by += [fr[o, 1], fr[h, 1], None]; bz += [fr[o, 2], fr[h, 2], None]
        traces.append(go.Scatter3d(x=bx, y=by, z=bz, mode="lines", line=dict(color="grey", width=2), name="bonds"))
        ax_, ay_, az_ = [], [], []
        for i in range(len(fr)):
            ax_ += [fr[i, 0], fr[i, 0] + 6 * dq[i, 0], None]
            ay_ += [fr[i, 1], fr[i, 1] + 6 * dq[i, 1], None]
            az_ += [fr[i, 2], fr[i, 2] + 6 * dq[i, 2], None]
        traces.append(go.Scatter3d(x=ax_, y=ay_, z=az_, mode="lines", line=dict(color="black", width=1),
                                   name=f"displacement over {args.stride_fs:.0f} fs (x6)"))
        fig = go.Figure(traces)
        fig.update_layout(scene=dict(aspectmode="cube"), title="64-water box with one-stride displacements")
        fig.write_html(os.path.join(d, "vis_water.html"))
        print("interactive view written", os.path.join(d, "vis_water.html"))
    except ImportError:
        print("plotly not installed; skipped interactive view (uv pip install plotly)")
    print("figures written to", d)


if __name__ == "__main__":
    main()
