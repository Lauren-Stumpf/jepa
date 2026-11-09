"""
Tests for train/pretrain_jepa.py, on a tiny synthetic dataset (CPU is fine).

  python checks/test_pretrain_jepa.py

Checks: config loading gives the model's cutoff and masses; the future state is
recovered from the stored mass-scaled targets; the same O(3) operation is applied to
both frames of a pair by default, and independent ones for the ablation; encoder is
deterministic and its features are not rotation invariant (as expected for PET); VICReg terms behave (collapsed batch gives a large variance loss,
decorrelated unit-variance batch gives near zero); the loss falls over a few steps on
a small batch; the saved checkpoint is loadable by metatrain in "finetune" context
and contains the trained encoder weights; a two-epoch end-to-end run resumes from its
state file.
"""

import json
import os
import subprocess
import sys
import tempfile

import ase.io
import numpy as np
import torch
from ase.build import molecule

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.pretrain_jepa import (PairData, Predictor, build_model, encode, jepa_loss,  # noqa: E402
                                 load_config, make_batch, vicreg_terms)

CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "flashmd_organic8_scratch.yaml")
torch.set_default_dtype(torch.float32)


def synthetic_file(path, n_frames, masses, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_frames):
        a = molecule(["CH4", "C2H6", "CH3OH", "NH3"][i % 4])
        a.set_cell([12.0, 12.0, 12.0]); a.pbc = True; a.center()
        a.rattle(0.05, seed=seed + i)
        m = np.array([masses[int(z)] for z in a.numbers]); sm = np.sqrt(m)[:, None]
        p = rng.normal(0, 0.3, (len(a), 3)) * sm            # thermal-ish momenta
        dq = rng.normal(0, 0.05, (len(a), 3)) / sm
        p_fut = p + rng.normal(0, 0.05, (len(a), 3)) * sm
        a.set_momenta(p)
        a.arrays["future_positions"] = a.get_positions() + dq * sm   # stored mass-scaled, as in the data
        a.arrays["future_momenta"] = p_fut / sm
        frames.append(a)
    ase.io.write(path, frames, format="extxyz")
    return frames


def test_config_and_data():
    cfg, hypers, masses, timestep, info = load_config(CONFIG)
    assert hypers["cutoff"] == 4.5 and masses[1] == 1.00794 and timestep == 8.0
    assert info.targets == {} and info.atomic_types == [1, 6, 7, 8]   # no targets: heads and scaler are added at fine-tune
    with tempfile.TemporaryDirectory() as d:
        frames = synthetic_file(os.path.join(d, "t.xyz"), 4, masses)
        data = PairData(os.path.join(d, "t.xyz"), masses)
        a = frames[1]
        m = np.array([masses[int(z)] for z in a.numbers]); sm = np.sqrt(m)[:, None]
        q_fut = a.get_positions() + (a.arrays["future_positions"] - a.get_positions()) / sm
        assert np.allclose(data.frames[1][3].numpy(), q_fut, atol=1e-6)
        assert np.allclose(data.frames[1][4].numpy(), a.arrays["future_momenta"] * sm, atol=1e-6)
    return cfg, hypers, masses, timestep, info


def test_augmentation_and_invariance(hypers, masses, timestep, info):
    model = build_model(hypers, info, masses, timestep).eval()
    with tempfile.TemporaryDirectory() as d:
        synthetic_file(os.path.join(d, "t.xyz"), 6, masses)
        data = PairData(os.path.join(d, "t.xyz"), masses)
    torch.manual_seed(1)
    now0, fut0 = make_batch(data.frames[:3], model, torch.float32, "cpu", augment=False)
    torch.manual_seed(1)
    now1, fut1 = make_batch(data.frames[:3], model, torch.float32, "cpu", augment=True)   # shared (default)
    for a0, a1, b0, b1 in zip(now0, now1, fut0, fut1):
        # shared: the same orthogonal matrix R takes a0 -> a1 and b0 -> b1 (positions and momenta)
        R = torch.linalg.lstsq(a0.positions, a1.positions).solution.T   # a1 = a0 @ R.T
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-4)
        assert torch.allclose(b0.positions @ R.T, b1.positions, atol=1e-4)
        p0 = a0.get_data("momentum").block().values.squeeze(-1)
        p1 = a1.get_data("momentum").block().values.squeeze(-1)
        assert torch.allclose(p0 @ R.T, p1, atol=1e-4), "momenta not rotated with positions"
    torch.manual_seed(2)
    now2, fut2 = make_batch(data.frames[:3], model, torch.float32, "cpu", augment=True, shared_rotation=False)   # ablation
    for a0, a2, b0, b2 in zip(now0, now2, fut0, fut2):
        Ra = torch.linalg.lstsq(a0.positions, a2.positions).solution.T
        Rb = torch.linalg.lstsq(b0.positions, b2.positions).solution.T
        assert torch.allclose(Ra @ Ra.T, torch.eye(3), atol=1e-4) and torch.allclose(Rb @ Rb.T, torch.eye(3), atol=1e-4)
        assert not torch.allclose(Ra, Rb, atol=1e-3), "independent rotations expected for the two frames"
    with torch.no_grad():
        z0 = encode(model, now0); z0b = encode(model, now0); z1 = encode(model, now1)
    assert torch.allclose(z0, z0b), "encoder is not deterministic"
    # PET features are NOT rotation invariant (invariance is only learned through
    # augmentation), which is why the two frames of a pair get independent rotations
    rel = ((z0 - z1).norm() / z0.norm()).item()
    assert 1e-3 < rel < 2.0, f"unexpected feature change under rotation for an untrained encoder: {rel}"
    assert z0.shape[1] >= 256, z0.shape   # 640 with our hypers (node features of 3 layers + edge features)
    return model, data


def test_vicreg():
    collapsed = torch.zeros(64, 8) + torch.randn(1, 8)
    v, c, d = vicreg_terms(collapsed)
    assert v.item() > 0.9 and d["std"] < 0.1 and d["dead_frac"] == 1.0
    good = torch.randn(4000, 8)
    v, c, d = vicreg_terms(good)
    assert v.item() < 0.05 and c.item() < 0.05 and abs(d["std"] - 1) < 0.05 and d["eff_rank"] > 7.5
    dup = torch.randn(4000, 4).repeat(1, 2)   # every dimension duplicated: covariance penalty must be large
    v, c, d = vicreg_terms(dup)
    assert c.item() > 0.5 and d["eff_rank"] < 4.5


def test_predictor_variants():
    z = torch.randn(10, 16) * 5 + 2
    pred = Predictor(16, 32, variant="residual")
    assert torch.allclose(pred(z), z), "residual predictor must start as an exact copy"
    for v in Predictor.VARIANTS:
        assert Predictor(16, 32, variant=v)(z).shape == z.shape
    from train.pretrain_jepa import normalise_embeddings
    n = normalise_embeddings(z)
    assert torch.allclose(n.mean(-1), torch.zeros(10), atol=1e-5) and torch.allclose(n.std(-1, unbiased=False), torch.ones(10), atol=1e-3)


def test_relative_loss():
    pred = Predictor(16, 32, variant="residual")           # starts as an exact copy
    z_now = torch.randn(50, 16); z_fut = z_now + 0.01 * torch.randn(50, 16)
    loss, parts = jepa_loss(z_now, z_fut, pred, 1.0, 0.0, 0.0, relative=True)
    assert abs(loss.item() - 1.0) < 1e-4, loss.item()       # copying scores exactly 1
    assert abs(parts["inv_ratio"] - 1.0) < 1e-4              # and the logged ratio says so (not divided twice)
    # the denominator is not detached: shrinking the change must not lower the loss by itself
    z_now2 = z_now.clone().requires_grad_(True); z_fut2 = (z_now2 + 0.01 * torch.randn(50, 16)).detach()
    loss2, _ = jepa_loss(z_now2, z_fut2 + 0 * z_now2, pred, 1.0, 0.0, 0.0, relative=True)
    assert loss2.requires_grad
    half = lambda z: z + 0.5 * (z_fut - z)                  # a predictor that explains half the change
    inv = torch.nn.functional.mse_loss(half(z_now), z_fut) / torch.nn.functional.mse_loss(z_now, z_fut)
    assert abs(inv.item() - 0.25) < 1e-6                    # half the change left -> a quarter of the squared error


def test_loss_decreases(model, data):
    model.train()
    now, fut = make_batch(data.frames[:6], model, torch.float32, "cpu", augment=False)
    pred = Predictor(encode(model, now).shape[1], 128)
    opt = torch.optim.Adam(list(model.parameters()) + list(pred.parameters()), lr=1e-3)
    losses = []
    for _ in range(12):
        loss, parts = jepa_loss(encode(model, now), encode(model, fut), pred, 25.0, 25.0, 1.0)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.9, losses
    assert all(np.isfinite(losses))


def test_end_to_end_and_checkpoint(masses):
    script = os.path.join(os.path.dirname(__file__), "..", "train", "pretrain_jepa.py")
    with tempfile.TemporaryDirectory() as d:
        synthetic_file(os.path.join(d, "train.xyz"), 12, masses, seed=0)
        synthetic_file(os.path.join(d, "val.xyz"), 4, masses, seed=7)
        out = os.path.join(d, "run")
        common = [sys.executable, script, CONFIG, "--train", os.path.join(d, "train.xyz"), "--val", os.path.join(d, "val.xyz"),
                  "--out", out, "--batch_size", "4", "--pred_hidden", "64"]
        for v in ("plain", "residual"):   # the other variants share the code path
            r = subprocess.run(common[:-2] + ["--out", out + "_" + v, "--epochs", "1", "--predictor", v, "--pull_every", "1"],
                               capture_output=True, text=True)
            assert r.returncode == 0, v + r.stdout[-2000:] + r.stderr[-3000:]
            pulls = json.load(open(os.path.join(out + "_" + v, "pulls.json")))
            assert pulls and all(k in pulls[0] for k in ("pull_inv", "pull_var", "pull_cov", "pull_cos_inv_reg")), pulls[:1]
        r = subprocess.run(common + ["--epochs", "1"], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
        st1 = torch.load(os.path.join(out, "state.pt"), map_location="cpu", weights_only=False)
        assert st1["epoch"] == 0
        r = subprocess.run(common + ["--epochs", "2"], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
        assert "resumed from epoch 0" in r.stdout
        st2 = torch.load(os.path.join(out, "state.pt"), map_location="cpu", weights_only=False)
        assert st2["epoch"] == 1 and len(st2["history"]) == 2
        # the metatrain checkpoint loads in finetune context and carries the trained encoder
        from metatrain.utils.io import model_from_checkpoint
        ckpt = torch.load(os.path.join(out, "pretrained.ckpt"), map_location="cpu", weights_only=False)
        loaded = model_from_checkpoint(ckpt, context="finetune")
        key = next(k for k in st2["model"] if k.startswith("gnn_layers.0") and st2["model"][k].ndim == 2)
        assert torch.allclose(loaded.state_dict()[key], st2["model"][key])
        assert float(loaded.timestep) == 8.0 and abs(float(loaded.masses[1]) - masses[1]) < 1e-5
        # fine-tuning must ADD the targets and FIT the scaler: with the full dataset info the
        # restarted model must report new targets and the scaler must list them as new outputs
        from omegaconf import OmegaConf
        from metatrain.utils.data import DatasetInfo
        from metatrain.utils.data.target_info import get_generic_target_info
        from metatrain.utils.omegaconf import expand_dataset_config
        import yaml
        cfg = yaml.safe_load(open(CONFIG))
        topts = expand_dataset_config(OmegaConf.create(cfg["training_set"]))[0]
        full = DatasetInfo(length_unit="A", atomic_types=[1, 6, 7, 8],
                           targets={n: get_generic_target_info(n, t) for n, t in topts["targets"].items()})
        restarted = loaded.restart(full)
        assert restarted.has_new_targets, "fine-tune would not add the heads"
        assert sorted(restarted.scaler.new_outputs) == ["momentum", "position"], "scaler would not be fitted at fine-tune"
        assert torch.allclose(restarted.state_dict()[key], st2["model"][key]), "encoder weights lost in restart"


if __name__ == "__main__":
    cfg, hypers, masses, timestep, info = test_config_and_data(); print("ok config and data")
    model, data = test_augmentation_and_invariance(hypers, masses, timestep, info); print("ok augmentation and invariance")
    test_vicreg(); print("ok vicreg terms")
    test_predictor_variants(); print("ok predictor variants")
    test_relative_loss(); print("ok relative loss")
    test_loss_decreases(model, data); print("ok loss decreases")
    test_end_to_end_and_checkpoint(masses); print("ok end to end, resume, checkpoint loads for finetune")
    print("all 7 tests passed")
