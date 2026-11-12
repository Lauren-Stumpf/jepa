"""
JEPA pretraining of the FlashMD encoder, with a VICReg regulariser (no momentum encoder).

What is learned. The FlashMD network (metatrain's PET backbone) maps every atom, given
its neighbourhood and its momentum, to a feature vector: metatrain exposes this as the
"feature" output (640 numbers per atom with our hypers). The
same network is applied to the current frame (q, p) and to the frame one stride later
(q', p'). A small predictor network has to turn each atom's current feature vector into
its future feature vector. Nothing in position space is predicted.

Loss (VICReg, Bardes, Ponce, LeCun 2022), on per-atom vectors over a batch:
  invariance   mean squared error between predicted and future feature vectors
  variance     for both the current and the future feature vectors, a hinge that pushes
               the standard deviation of every feature dimension (over the atoms in the
               batch) up to 1, which stops all atoms collapsing to one vector
  covariance   for both, the squared off-diagonal entries of the feature covariance,
               which stops dimensions duplicating each other
Gradients flow through both branches; there is no exponential-moving-average target
encoder and no stop-gradient.

Augmentation: the same random rotation or improper rotation is applied to both frames
of a pair (positions, cell and momenta), as in FlashMD training. Shared rather than
independent because the FlashMD heads are linear maps from the features to Cartesian
vectors, so the features must keep orientation; independent rotations would train the
encoder to discard it. The possible shortcut (orientation itself as a predictable
feature) is three numbers and cannot satisfy the variance and covariance terms over
640 dimensions; the effective-rank diagnostic would show it. --independent_rotations
runs the alternative as an ablation.

Output: a metatrain checkpoint of the FlashMD model (encoder pretrained, prediction
heads untouched) that `mtt train` can start from through `finetune: {read_from: ...}`,
so fine-tuning is exactly the baseline's training with a different starting point.

Resumes from its own state file if interrupted (state.pt in the output directory).

  python train/pretrain_jepa.py configs/flashmd_organic8_scratch.yaml \
      --train data/organic8/train.xyz --val data/organic8/val.xyz \
      --out results/runs/jepa_pretrain_organic8 --epochs 20
"""

import argparse
import json
import math
import os
import time

import ase.io
import numpy as np
import torch
import yaml
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System
from metatomic.torch.o3 import random_transformations
from omegaconf import OmegaConf

from metatrain.experimental.flashmd import __model__ as FlashMD
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists
from metatrain.utils.omegaconf import expand_dataset_config


# ----------------------------------------------------------------------------- config

def load_config(path):
    cfg = yaml.safe_load(open(path))
    arch = cfg["architecture"]
    model_hypers = dict(get_default_hypers("experimental.flashmd")["model"])
    model_hypers.update(arch.get("model", {}))
    masses = {int(k): float(v) for k, v in arch["training"]["masses"].items()}
    timestep = float(arch["training"]["timestep"])
    train_opts = expand_dataset_config(OmegaConf.create(cfg["training_set"]))[0]
    # The pretraining model is built with NO targets. If it carried the position and
    # momentum targets, its checkpoint would also carry an unfitted target scaler (all
    # ones), and metatrain's fine-tuning would keep it instead of fitting it to the data
    # as the baseline does: the two arms would then minimise different losses. With no
    # targets, fine-tuning adds the heads and fits the scaler exactly as from scratch.
    dataset_info = DatasetInfo(length_unit=str(train_opts["systems"]["length_unit"]),
                               atomic_types=sorted(arch["atomic_types"]), targets={})
    return cfg, model_hypers, masses, timestep, dataset_info


def build_model(model_hypers, dataset_info, masses, timestep):
    model = FlashMD(model_hypers, dataset_info)
    model.set_masses(masses)
    model.set_timestep(timestep)
    return model


# ------------------------------------------------------------------------------- data

class PairData:
    """All frames of one xyz file as plain tensors: (types, q, p, q', p', cell, pbc).
    Stored targets are mass-scaled (data/prepare_organic.py); the raw future state is
    recovered here with the config masses."""

    def __init__(self, path, masses, max_frames=0):
        self.frames = []
        for i, a in enumerate(ase.io.iread(path, index=":", format="extxyz")):
            if max_frames and i >= max_frames:
                break
            m = np.array([masses[int(z)] for z in a.numbers])
            sm = np.sqrt(m)[:, None]
            q = a.get_positions()
            q_fut = q + (a.arrays["future_positions"] - q) / sm
            p_fut = a.arrays["future_momenta"] * sm
            self.frames.append((
                torch.tensor(a.numbers, dtype=torch.int32),
                torch.tensor(q), torch.tensor(a.get_momenta()),
                torch.tensor(q_fut), torch.tensor(p_fut),
                torch.tensor(a.cell.array), torch.tensor(a.pbc),
            ))

    def __len__(self):
        return len(self.frames)


def make_system(types, q, p, cell, pbc, dtype, device):
    system = System(types=types.to(device), positions=q.to(dtype=dtype, device=device),
                    cell=cell.to(dtype=dtype, device=device), pbc=pbc.to(device))
    n = len(types)
    samples = Labels(names=["system", "atom"],
                     values=torch.stack([torch.zeros(n, dtype=torch.int32), torch.arange(n, dtype=torch.int32)], 1).to(device))
    xyz = Labels(names="xyz", values=torch.arange(3, dtype=torch.int32).reshape(-1, 1).to(device))
    block = TensorBlock(values=p.to(dtype=dtype, device=device).unsqueeze(-1), samples=samples, components=[xyz],
                        properties=Labels(names=["momentum"], values=torch.zeros((1, 1), dtype=torch.int32).to(device)))
    system.add_data("momentum", TensorMap(keys=Labels.single().to(device), blocks=[block]))
    return system


def make_batch(frames, model, dtype, device, augment, shared_rotation=True):
    """Two lists of Systems (now, future). With augment, each pair gets a random O(3)
    operation applied to both frames; with shared_rotation=False the two frames get
    independent operations."""
    now = [make_system(t, q, p, c, b, dtype, device) for t, q, p, _, _, c, b in frames]
    fut = [make_system(t, qf, pf, c, b, dtype, device) for t, _, _, qf, pf, c, b in frames]
    if augment:
        ops = random_transformations(len(frames), 0, device=torch.device(device), dtype=dtype, add_inversions=True)
        now = ops.transform_systems(now)
        if not shared_rotation:
            ops = random_transformations(len(frames), 0, device=torch.device(device), dtype=dtype, add_inversions=True)
        fut = ops.transform_systems(fut)
    nl = model.requested_neighbor_lists()
    now = [get_system_with_neighbor_lists(s, nl) for s in now]
    fut = [get_system_with_neighbor_lists(s, nl) for s in fut]
    return now, fut


# ------------------------------------------------------------------------------ model

FEATURE_OUTPUT = {"feature": ModelOutput(sample_kind="atom")}
VAL_SHUFFLE_SEED = 12345   # fixed order of validation pairs, shared by all runs


def encode(model, systems):
    """Per-atom feature vectors [n_atoms_in_batch, d].

    The encoder is always run in training mode: in evaluation mode metatrain's FlashMD
    applies its position additive to the outputs even when only "feature" was requested
    and fails with KeyError 'position' (metatrain 2026.4.1). Training mode changes nothing
    for the feature output (no dropout or batch normalisation in PET); gradients are
    still controlled by torch.no_grad where needed."""
    was_training = model.training
    model.train(True)
    try:
        return model(systems, FEATURE_OUTPUT)["feature"].block().values
    finally:
        model.train(was_training)


class Projector(torch.nn.Module):
    """Optional small network between the encoder and the loss (VICReg's "expander").
    With it, the variance and covariance terms act on the projector's output rather
    than on the encoder's own features, so the regulariser cannot reshape the encoder
    directly; the encoder is only shaped through what the projector needs."""

    def __init__(self, d, out):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(d), torch.nn.Linear(d, out), torch.nn.GELU(), torch.nn.Linear(out, out))

    def forward(self, z):
        return self.net(z)


class Predictor(torch.nn.Module):
    """Maps the current embedding to the predicted future embedding. Variants:
      plain        guess = f(now); no normalisation anywhere
      norm_input   guess = f(normalise(now)); target NOT normalised (the first run's design:
                   the predictor has to undo the normalisation, which makes copying hard)
      norm_target  one normalising step on the encoder output for BOTH frames, applied
                   outside the predictor (see normalise_embeddings); here guess = f(now)
      residual     guess = now + f(now), with f's last layer started at zero so the
                   predictor begins as an exact copy and only learns the change"""

    VARIANTS = ("plain", "norm_input", "norm_target", "residual")

    def __init__(self, d, hidden, variant="norm_target"):
        super().__init__()
        assert variant in self.VARIANTS, variant
        self.variant = variant
        layers = [torch.nn.Linear(d, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
                  torch.nn.Linear(hidden, d)]
        if variant in ("norm_input", "residual"):
            layers = [torch.nn.LayerNorm(d)] + layers
        self.net = torch.nn.Sequential(*layers)
        if variant == "residual":
            torch.nn.init.zeros_(self.net[-1].weight)
            torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, z):
        return z + self.net(z) if self.variant == "residual" else self.net(z)


def normalise_embeddings(z):
    """Per atom: mean 0 and spread 1 over the 640 numbers (a fixed LayerNorm without learnt scale)."""
    return torch.nn.functional.layer_norm(z, z.shape[-1:])


def vicreg_terms(z):
    """(variance hinge, covariance penalty, diagnostics) of a [n, d] batch of vectors."""
    z = z - z.mean(0, keepdim=True)
    std = torch.sqrt(z.var(0, unbiased=False) + 1e-4)
    var_loss = torch.relu(1.0 - std).mean()
    n, d = z.shape
    cov = (z.T @ z) / max(n - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    cov_loss = (off ** 2).sum() / d
    with torch.no_grad():
        # effective rank: exp of the entropy of the normalised covariance eigenvalues
        ev = torch.linalg.eigvalsh(cov.double()).clamp(min=0)
        pr = ev / ev.sum().clamp(min=1e-12)
        eff_rank = torch.exp(-(pr * torch.log(pr.clamp(min=1e-12))).sum()).item()
        diag = {"std": std.mean().item(), "std_min": std.min().item(),
                "dead_frac": (std < 0.5).float().mean().item(), "eff_rank": eff_rank}
    return var_loss, cov_loss, diag


def term_pulls(z_now, z_fut, pred, model, w_inv, w_var, w_cov, relative=False):
    """How hard each weighted term pushes on the ENCODER's weights: the size of its
    gradient over the encoder parameters, and the cosine between the prediction push
    and the regulariser push (negative means they pull in opposite directions)."""
    enc_params = [p for p in model.parameters() if p.requires_grad]
    z_hat = pred(z_now)
    inv = torch.nn.functional.mse_loss(z_hat, z_fut)
    if relative:
        inv = inv / torch.nn.functional.mse_loss(z_now, z_fut).clamp(min=1e-12)
    inv = w_inv * inv
    v1, c1, _ = vicreg_terms(z_now)
    v2, c2, _ = vicreg_terms(z_fut)
    var = w_var * 0.5 * (v1 + v2)
    cov = w_cov * 0.5 * (c1 + c2)
    out = {}
    grads = {}
    for name, term in (("inv", inv), ("var", var), ("cov", cov)):
        g = torch.autograd.grad(term, enc_params, retain_graph=True, allow_unused=True)
        flat = torch.cat([x.reshape(-1) for x in g if x is not None])
        grads[name] = flat
        out[f"pull_{name}"] = flat.norm().item()
    reg = grads["var"] + grads["cov"]
    out["pull_cos_inv_reg"] = torch.nn.functional.cosine_similarity(grads["inv"], reg, dim=0).item()
    tot = out["pull_inv"] + out["pull_var"] + out["pull_cov"]
    out["pull_inv_share"] = out["pull_inv"] / max(tot, 1e-12)
    return out


def jepa_loss(z_now, z_fut, pred, w_inv, w_var, w_cov, relative=False):
    z_hat = pred(z_now)
    inv = torch.nn.functional.mse_loss(z_hat, z_fut)
    trivial = torch.nn.functional.mse_loss(z_now, z_fut)   # the error of plain copying
    if relative:
        # prediction error measured against the size of the actual change, so copying scores 1
        # and explaining half the change scores 0.25 whatever the absolute size of the change.
        # The denominator is NOT detached: detaching it gives the encoder a direct incentive to
        # make the two frames identical (delete everything that changes in time), growing as
        # the change shrinks (independent review, 9 Nov). Undetached, the ratio is scale-free
        # and the only way to lower it is to make the change predictable.
        inv = inv / trivial.clamp(min=1e-12)
    v1, c1, d1 = vicreg_terms(z_now)
    v2, c2, _ = vicreg_terms(z_fut)
    var = 0.5 * (v1 + v2)
    cov = 0.5 * (c1 + c2)
    total = w_inv * inv + w_var * var + w_cov * cov
    # the ratio to the trivial prediction "future feature = current feature" is the headline
    # diagnostic: near 1 means the predictor has learned nothing about motion
    ratio = inv if relative else inv / trivial.detach().clamp(min=1e-12)
    return total, {"inv": inv.item(), "var": var.item(), "cov": cov.item(), "trivial_inv": trivial.item(),
                   "inv_ratio": ratio.item(),
                   "std_now": d1["std"], "std_min": d1["std_min"], "dead_frac": d1["dead_frac"], "eff_rank": d1["eff_rank"]}


# ------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64,
                    help="larger than the baseline's 16 because the VICReg terms are estimated over the atoms of a batch")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup_fraction", type=float, default=0.01)
    ap.add_argument("--w_inv", type=float, default=25.0)
    ap.add_argument("--w_var", type=float, default=25.0)
    ap.add_argument("--w_cov", type=float, default=1.0)
    ap.add_argument("--pred_hidden", type=int, default=1024)
    ap.add_argument("--predictor", default="norm_target", choices=Predictor.VARIANTS,
                    help="plain | norm_input (first run's design) | norm_target (I-JEPA style, default) | residual")
    ap.add_argument("--residual_predictor", action="store_true", help="same as --predictor residual")
    ap.add_argument("--relative_loss", action="store_true",
                    help="prediction error divided by the error of plain copying (so it is inv/trivial itself)")
    ap.add_argument("--pull_every", type=int, default=200,
                    help="every N steps, log how hard each loss term pushes on the encoder (0 disables)")
    ap.add_argument("--projector_dim", type=int, default=0,
                    help="0: regulariser and prediction act on the encoder features directly; "
                         ">0: on the output of a projector network of this width (VICReg expander)")
    ap.add_argument("--max_frames", type=int, default=0, help="use only the first N frames of each file (smoke tests)")
    ap.add_argument("--no_augment", action="store_true")
    ap.add_argument("--independent_rotations", action="store_true",
                    help="ablation: independent random rotations for the two frames of a pair (default: shared)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", default="", help="wandb project name; empty disables")
    ap.add_argument("--wandb_group", default=None, help="wandb group: the experiment id (e.g. E06)")
    ap.add_argument("--wandb_job_type", default=None, help="wandb job type: the arm (e.g. pretrain-regonly)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    cfg, model_hypers, masses, timestep, dataset_info = load_config(args.config)
    model = build_model(model_hypers, dataset_info, masses, timestep).to(device=device, dtype=dtype)
    for additive in model.additive_models:   # metatrain keeps these in float64
        additive.to(dtype=torch.float64)
    model.scaler.to(dtype=torch.float64)

    t0 = time.time()
    train = PairData(args.train, masses, args.max_frames)
    val = PairData(args.val, masses, args.max_frames)
    # The validation file is grouped by molecule (or water run), so batches taken in file
    # order hold one molecule each, and the per-batch diagnostics (variance, covariance,
    # effective rank, dead fraction) then measure spread within one molecule, not across
    # atoms in general as the shuffled training batches do. Shuffle once with a FIXED seed
    # (independent of --seed): every epoch, variant and seed then sees the same mixed batches.
    val_order = np.random.default_rng(VAL_SHUFFLE_SEED).permutation(len(val.frames))
    val.frames = [val.frames[i] for i in val_order]
    print(f"read {len(train)} training and {len(val)} validation pairs in {time.time() - t0:.0f} s", flush=True)

    with torch.no_grad():
        now, _ = make_batch(train.frames[:2], model, dtype, device, augment=False)
        d = encode(model, now).shape[1]
    if args.projector_dim > 0:
        proj = Projector(d, args.projector_dim).to(device=device, dtype=dtype)
        d_loss = args.projector_dim
    else:
        proj = torch.nn.Identity()
        d_loss = d
    variant = "residual" if args.residual_predictor else args.predictor
    pred = Predictor(d_loss, args.pred_hidden, variant=variant).to(device=device, dtype=dtype)
    # for norm_target the normalising step sits at the encoder output, for both frames
    embed = normalise_embeddings if variant == "norm_target" else (lambda z: z)
    print(f"predictor variant: {variant}; projector: {args.projector_dim or 'none'}; "
          f"prediction loss: {'relative to copying' if args.relative_loss else 'absolute'}", flush=True)
    print(f"feature dimension {d}; encoder {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M, "
          f"predictor {sum(p.numel() for p in pred.parameters()) / 1e6:.1f}M parameters", flush=True)

    params = list(model.parameters()) + list(pred.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warm = max(1, int(args.warmup_fraction * total_steps))

    def lr_at(step):
        if step < warm:
            return args.lr * (step + 1) / warm
        frac = (step - warm) / max(1, total_steps - warm)
        return args.lr * 0.5 * (1 + math.cos(math.pi * frac))

    state_path = os.path.join(args.out, "state.pt")
    start_epoch, step = 0, 0
    history = []
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); pred.load_state_dict(st["pred"]); opt.load_state_dict(st["opt"])
        if "proj" in st:
            proj.load_state_dict(st["proj"])
        start_epoch, step, history = st["epoch"] + 1, st["step"], st["history"]
        if st.get("epochs_planned", args.epochs) != args.epochs:
            print(f"WARNING: resuming with --epochs {args.epochs} but the run was planned for "
                  f"{st['epochs_planned']}; the learning-rate schedule will not match", flush=True)
        print(f"resumed from epoch {st['epoch']}", flush=True)

    run = None
    if args.wandb:
        import wandb
        run_name = os.path.basename(args.out.rstrip("/"))
        # one id per pretraining, kept in a file so a resumed run continues the same wandb run; a
        # new id is made if the file is missing (after scripts/pretrain_supervisor.sh deleted an
        # abandoned run, whose id wandb will not accept again)
        id_file = os.path.join(args.out, "wandb_run_id")
        if os.path.exists(id_file):
            run_id = open(id_file).read().strip()
        else:
            import uuid
            run_id = f"{run_name}-{uuid.uuid4().hex[:6]}"
            with open(id_file, "w") as f:
                f.write(run_id)
        # one wandb run per pretraining (same id on resume); its local files live in the run folder,
        # so scripts/wandb_delete_runs.py can find and remove the run if it is abandoned
        run = wandb.init(project=args.wandb, name=run_name, id=run_id, config=vars(args), resume="allow",
                         group=args.wandb_group, job_type=args.wandb_job_type, dir=args.out)

    pull_log = []
    rng = np.random.default_rng(args.seed + start_epoch)
    for epoch in range(start_epoch, args.epochs):
        model.train(); pred.train(); proj.train()
        order = rng.permutation(len(train))
        agg = {}
        t_ep = time.time()
        for b in range(steps_per_epoch):
            idx = order[b * args.batch_size:(b + 1) * args.batch_size]
            now, fut = make_batch([train.frames[i] for i in idx], model, dtype, device,
                                  augment=not args.no_augment, shared_rotation=not args.independent_rotations)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            z_now = encode(model, now)
            z_fut = encode(model, fut)
            e_now, e_fut = embed(proj(z_now)), embed(proj(z_fut))
            loss, parts = jepa_loss(e_now, e_fut, pred, args.w_inv, args.w_var, args.w_cov, args.relative_loss)
            if b % 200 == 0 and args.projector_dim > 0:   # the encoder's own features, as well as the loss space
                _, _, de = vicreg_terms(z_now.detach())
                parts.update({"enc_rank": de["eff_rank"], "enc_dead": de["dead_frac"]})
            if args.pull_every and b % args.pull_every == 0:
                if True:
                    pulls = term_pulls(e_now, e_fut, pred, model, args.w_inv, args.w_var, args.w_cov, args.relative_loss)
                    parts.update(pulls)
                    pull_log.append({"epoch": epoch, "step": b, **pulls})
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            parts["loss"] = loss.item()
            for k, v in parts.items():
                if not (k.startswith("enc_") or k.startswith("pull_")):
                    agg[k] = agg.get(k, 0.0) + v
            if b % 200 == 0:
                pull_txt = (f" | pull on encoder: inv {parts['pull_inv']:.3g} var {parts['pull_var']:.3g} cov {parts['pull_cov']:.3g} "
                            f"(inv share {parts['pull_inv_share']:.2f}, cos {parts['pull_cos_inv_reg']:+.2f})") if "pull_inv" in parts else ""
                print(f"  epoch {epoch} step {b}/{steps_per_epoch} loss {loss.item():.4f} inv/trivial {parts['inv_ratio']:.3f} "
                      f"var {parts['var']:.4f} cov {parts['cov']:.4f} std_min {parts['std_min']:.3f} "
                      f"dead {parts['dead_frac']:.2f} rank {parts['eff_rank']:.0f}{pull_txt}", flush=True)
        train_log = {f"train_{k}": v / steps_per_epoch for k, v in agg.items()}

        model.eval(); pred.eval(); proj.eval()
        vagg = {}
        nv = 0
        with torch.no_grad():
            for b in range(0, len(val), args.batch_size):
                now, fut = make_batch(val.frames[b:b + args.batch_size], model, dtype, device, augment=False)
                h_now = encode(model, now)
                loss, parts = jepa_loss(embed(proj(h_now)), embed(proj(encode(model, fut))), pred, args.w_inv, args.w_var, args.w_cov, args.relative_loss)
                _, _, de = vicreg_terms(h_now)
                parts.update({"enc_rank": de["eff_rank"], "enc_dead": de["dead_frac"]})
                parts["loss"] = loss.item()
                for k, v in parts.items():
                    vagg[k] = vagg.get(k, 0.0) + v
                nv += 1
        val_log = {f"val_{k}": v / max(nv, 1) for k, v in vagg.items()}
        rec = {"epoch": epoch, "lr": lr_at(step - 1), "seconds": time.time() - t_ep, **train_log, **val_log}
        history.append(rec)
        if run is not None and pull_log:
            run.log({k: v for k, v in pull_log[-1].items() if k.startswith("pull_")}, step=epoch, commit=False)
        print(f"epoch {epoch}: train loss {rec['train_loss']:.4f} | val loss {rec['val_loss']:.4f} "
              f"inv/trivial {rec['val_inv_ratio']:.3f} var {rec['val_var']:.4f} cov {rec['val_cov']:.4f} "
              f"std_min {rec['val_std_min']:.3f} dead {rec['val_dead_frac']:.2f} rank {rec['val_eff_rank']:.0f} "
              f"| encoder features: dead {rec['val_enc_dead']:.2f} rank {rec['val_enc_rank']:.0f} "
              f"| {rec['seconds']:.0f} s", flush=True)
        if run is not None:
            run.log(rec, step=epoch)

        torch.save({"model": model.state_dict(), "pred": pred.state_dict(), "proj": proj.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch, "step": step, "history": history, "epochs_planned": args.epochs}, state_path)
        ckpt = model.get_checkpoint()
        ckpt["epoch"] = epoch
        ckpt["best_epoch"] = epoch
        torch.save(ckpt, os.path.join(args.out, "pretrained.ckpt"))
        torch.save(pred.state_dict(), os.path.join(args.out, "predictor.pt"))
        with open(os.path.join(args.out, "history.json"), "w") as f:
            json.dump(history, f, indent=1)
        with open(os.path.join(args.out, "pulls.json"), "w") as f:
            json.dump(pull_log, f, indent=1)

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"args": vars(args), "predictor_variant": variant, "feature_dim": d, "train_pairs": len(train), "val_pairs": len(val),
                   "masses": masses, "timestep_fs": timestep, "model_hypers": model_hypers,
                   "val_shuffle_seed": VAL_SHUFFLE_SEED}, f, indent=1)
    print("done:", os.path.join(args.out, "pretrained.ckpt"))
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
