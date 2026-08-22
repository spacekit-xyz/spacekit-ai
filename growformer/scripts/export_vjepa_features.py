#!/usr/bin/env python3
"""Export frozen V-JEPA (or mock teacher) features for Growformer Phase 3u.

Never fine-tunes the backbone. Writes a pinned feature bank + student projector
that Rust loads via `wm_vjepa.rs`.

Usage:
  # Offline mock teacher (no HF download; same JSON schema as real export)
  python3 scripts/export_vjepa_features.py --mode mock --out data/wm/vjepa_export_v1.json

  # Real Meta V-JEPA 2 (requires: pip install torch transformers)
  python3 scripts/export_vjepa_features.py --mode hf \\
      --model facebook/vjepa2-vitl-fpc64-256 \\
      --out data/wm/vjepa_export_v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np

VISION_SIDE = 8
VISION_PIXELS = 64
LATENT_DIM = 8


def render_visuomotor(gx: float, gy: float, ox: float, oy: float) -> np.ndarray:
    pix = np.zeros(VISION_PIXELS, dtype=np.float32)
    for iy in range(VISION_SIDE):
        for ix in range(VISION_SIDE):
            i = iy * VISION_SIDE + ix
            pix[i] = 0.18 if ix < VISION_SIDE // 2 else 0.02
    for iy in range(VISION_SIDE):
        pix[iy * VISION_SIDE + VISION_SIDE // 2] = 0.45

    def splat(x: float, y: float, val: float, rad: float) -> None:
        u = (x + 1.0) * 0.5 * (VISION_SIDE - 1)
        v = (1.0 - (y + 1.0) * 0.5) * (VISION_SIDE - 1)
        for iy in range(VISION_SIDE):
            for ix in range(VISION_SIDE):
                d = math.hypot(ix - u, iy - v)
                if d < rad:
                    i = iy * VISION_SIDE + ix
                    pix[i] = max(pix[i], val * (1.0 - d / rad))

    splat(gx, gy, 1.0, 1.5)
    splat(ox, oy, 0.9 if ox < 0 else 0.55, 1.7)
    return pix


def step_visuomotor(gx, gy, ox, oy, action: int):
    impulse = 0.08
    if action % 4 == 0:
        gx -= impulse
    elif action % 4 == 1:
        gx += impulse
    elif action % 4 == 2:
        gy += impulse
    else:
        gy -= impulse
    gx = float(np.clip(gx, -1, 1))
    gy = float(np.clip(gy, -1, 1))
    dx, dy = ox - gx, oy - gy
    dist = math.hypot(dx, dy)
    if dist < 0.22:
        soft = ox < 0.0
        gain = 0.55 if soft else 0.18
        px, py = [(-1, 0), (1, 0), (0, 1), (0, -1)][action % 4]
        nx = dx / dist if dist > 1e-4 else 1.0
        ny = dy / dist if dist > 1e-4 else 0.0
        ox += gain * impulse * (px + 0.25 * nx)
        oy += gain * impulse * (py + 0.25 * ny)
        if not soft:
            gx *= 0.85
            gy *= 0.85
    return gx, gy, float(np.clip(ox, -1, 1)), float(np.clip(oy, -1, 1))


def fingerprint_bytes(parts: list[bytes]) -> int:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return int.from_bytes(h.digest()[:8], "little")


def mock_teacher_embed(pixels: np.ndarray, seed: int = 0x4EFA) -> np.ndarray:
    """Deterministic frozen mock teacher (same schema as HF path)."""
    rng = np.random.RandomState(seed)
    # Fixed random projection + nonlinearities — never updated.
    w1 = rng.randn(128, VISION_PIXELS).astype(np.float32) * 0.15
    b1 = np.zeros(128, dtype=np.float32)
    w2 = rng.randn(256, 128).astype(np.float32) * 0.1
    b2 = np.zeros(256, dtype=np.float32)
    h = np.tanh(w1 @ pixels + b1)
    return np.tanh(w2 @ h + b2).astype(np.float32)


def hf_teacher_factory(model_id: str):
    import torch
    from transformers import AutoModel

    try:
        from transformers import AutoVideoProcessor as _Proc
    except ImportError:
        try:
            from transformers import AutoProcessor as _Proc
        except ImportError:
            from transformers import AutoImageProcessor as _Proc

    processor = _Proc.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def embed(pixels: np.ndarray) -> np.ndarray:
        # Upsample 8x8 → 256x256 RGB video clip (2 frames) for the processor.
        small = pixels.reshape(VISION_SIDE, VISION_SIDE)
        big = np.kron(small, np.ones((32, 32), dtype=np.float32))  # 256x256
        rgb = np.stack([big, big, big], axis=-1)  # HWC
        # Duplicate as a short clip
        video = np.stack([rgb, rgb], axis=0)  # T,H,W,C
        try:
            inputs = processor(video, return_tensors="pt")
        except TypeError:
            # Image processor path: pass a list of frames
            inputs = processor(images=[rgb, rgb], return_tensors="pt")
        if not isinstance(inputs, dict):
            inputs = dict(inputs)
        out = model(**inputs)
        # Pool last hidden if present
        if hasattr(out, "last_hidden_state"):
            z = out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            z = out.pooler_output.squeeze(0).cpu().numpy()
        else:
            # fallback: first tensor in outputs
            z = out[0].mean(dim=tuple(range(1, out[0].ndim))).squeeze().cpu().numpy()
        return z.astype(np.float32)

    return embed, 0  # dim filled later


def fit_projector(teacher_zs: np.ndarray, target_dim: int = LATENT_DIM):
    """Frozen linear map teacher → WM latent (fit once on export set, then freeze)."""
    # PCA-like via SVD
    x = teacher_zs - teacher_zs.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    w = vt[:target_dim].astype(np.float32)  # (8, D)
    b = (-w @ teacher_zs.mean(axis=0)).astype(np.float32)
    return w, b


def project(w: np.ndarray, b: np.ndarray, z: np.ndarray) -> np.ndarray:
    y = w @ z + b
    return np.tanh(y).astype(np.float32)


def fit_student(pixels: np.ndarray, targets: np.ndarray, steps: int = 400):
    """Distill frozen student MLP pixels→latent for Rust live encode (never trained later)."""
    rng = np.random.RandomState(0x57D01)
    hidden = 48
    w1 = rng.randn(hidden, VISION_PIXELS).astype(np.float32) * 0.2
    b1 = np.zeros(hidden, dtype=np.float32)
    w2 = rng.randn(LATENT_DIM, hidden).astype(np.float32) * 0.2
    b2 = np.zeros(LATENT_DIM, dtype=np.float32)
    lr = 0.05
    n = pixels.shape[0]
    for _ in range(steps):
        i = rng.randint(0, n)
        x, t = pixels[i], targets[i]
        h = np.tanh(w1 @ x + b1)
        y = np.tanh(w2 @ h + b2)
        dy = 2.0 * (y - t) / LATENT_DIM
        # d tanh
        dy *= 1.0 - y * y
        db2 = dy
        dw2 = np.outer(dy, h)
        dh = w2.T @ dy
        dh *= 1.0 - h * h
        b2 -= lr * db2
        w2 -= lr * dw2
        b1 -= lr * dh
        w1 -= lr * np.outer(dh, x)
    return w1, b1, w2, b2


def load_log_frames(path: Path):
    """Load JSONL visuomotor log (pixels / pixels_next / regime_left)."""
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mock", "hf"], default="mock")
    ap.add_argument("--model", default="facebook/vjepa2-vitl-fpc64-256")
    ap.add_argument("--out", type=Path, default=Path("data/wm/vjepa_export_v1.json"))
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--log",
        type=Path,
        default=None,
        help="JSONL visuomotor log (real-log path). If set, frames come from the log, not online sampling.",
    )
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    if args.mode == "hf":
        embed, _ = hf_teacher_factory(args.model)
        source = args.model
    else:
        embed = lambda p: mock_teacher_embed(p)
        source = "mock-vjepa-teacher-v1"

    frames = []
    teacher_list = []
    pix_list = []
    if args.log is not None:
        log_rows = load_log_frames(args.log)
        if args.n > 0:
            log_rows = log_rows[: args.n]
        for row in log_rows:
            pix = np.asarray(row["pixels"], dtype=np.float32)
            pix_n = np.asarray(row["pixels_next"], dtype=np.float32)
            zt = embed(pix)
            ztn = embed(pix_n)
            teacher_list.append(zt)
            pix_list.append(pix)
            frames.append(
                {
                    "pixels": pix.tolist(),
                    "pixels_next": pix_n.tolist(),
                    "regime_left": bool(row.get("regime_left", False)),
                    "z_teacher": zt.tolist(),
                    "z_teacher_next": ztn.tolist(),
                }
            )
        export_mode = "hf" if args.mode == "hf" else "real_log"
        source = f"{source}+log:{args.log.name}"
    else:
        for _ in range(args.n):
            gx, gy = rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8)
            ox = rng.uniform(-0.85, -0.05) if rng.rand() < 0.5 else rng.uniform(0.05, 0.85)
            oy = rng.uniform(-0.8, 0.8)
            action = int(rng.randint(0, 4))
            pix = render_visuomotor(gx, gy, ox, oy)
            gx2, gy2, ox2, oy2 = step_visuomotor(gx, gy, ox, oy, action)
            pix_n = render_visuomotor(gx2, gy2, ox2, oy2)
            zt = embed(pix)
            ztn = embed(pix_n)
            teacher_list.append(zt)
            pix_list.append(pix)
            frames.append(
                {
                    "pixels": pix.tolist(),
                    "pixels_next": pix_n.tolist(),
                    "regime_left": bool(ox < 0.0),
                    "z_teacher": zt.tolist(),
                    "z_teacher_next": ztn.tolist(),
                }
            )
        export_mode = args.mode

    teacher = np.stack(teacher_list, axis=0)
    jepa_dim = int(teacher.shape[1])
    w, b = fit_projector(teacher, LATENT_DIM)
    for fr in frames:
        fr["z"] = project(w, b, np.array(fr["z_teacher"], dtype=np.float32)).tolist()
        fr["z_next"] = project(w, b, np.array(fr["z_teacher_next"], dtype=np.float32)).tolist()
        del fr["z_teacher"]
        del fr["z_teacher_next"]

    targets = np.array([fr["z"] for fr in frames], dtype=np.float32)
    pixels = np.stack(pix_list, axis=0)
    sw1, sb1, sw2, sb2 = fit_student(pixels, targets)

    # Fingerprint over projector + student + source id (not frame contents alone)
    parts = [
        source.encode(),
        struct.pack("<I", jepa_dim),
        w.tobytes(),
        b.tobytes(),
        sw1.tobytes(),
        sb1.tobytes(),
        sw2.tobytes(),
        sb2.tobytes(),
    ]
    fp = fingerprint_bytes(parts)

    out = {
        "source_model": source,
        "export_mode": export_mode,
        "jepa_dim": jepa_dim,
        "latent_dim": LATENT_DIM,
        "fingerprint": fp,
        "projector_w": w.tolist(),
        "projector_b": b.tolist(),
        "student_w1": sw1.tolist(),
        "student_b1": sb1.tolist(),
        "student_w2": sw2.tolist(),
        "student_b2": sb2.tolist(),
        "note": "Frozen V-JEPA export bank. Kill gate: any gradient into projector/student/backbone. Use --log for real-log path.",
        "frames": frames,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print(f"Wrote {args.out} mode={export_mode} source={source} frames={len(frames)} fp={fp:#x}")


if __name__ == "__main__":
    main()
