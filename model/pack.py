"""The Doom machine as a Qwen3.5 model, kept as its nonzero weights: model/build/pack.npz.

The same model rv-tiny builds (qwenputer/chipformer/rv/tiny.py: Qwen3.5-4B's DeltaNet head shapes
and MLP width, the circuit's layers, the tied embedding of the chip tokens), for the Doom map
(QWENPUTER_MAP=doom): 300-odd layers, too many to hold as dense matrices. Each layer is emitted
(emit_layers with a sink), checked bf16-exact as rv-tiny's are, and kept as its nonzero entries,
input by input (CSC), as the stepper reads them. The embedding is built a family at a time from
the same per-family linearity rv-tiny uses, and kept sparse; the output head is the embedding's
columns at the code dims (the final norm zeroes every other dim).

Run: QWENPUTER_MAP=doom PYTHONPATH=qwenputer python model/pack.py
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from chipformer.q35.emit import check_bf16, emit_layers
from chipformer.q35.tiny import BASE, CODE_GAIN, pick_layout
from chipformer.rv import core
from chipformer.rv import spec as S

OUT = Path(__file__).resolve().parent / "build"
MATS = ("linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
        "linear_attn.in_proj_b.weight", "linear_attn.in_proj_a.weight",
        "linear_attn.out_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
        "mlp.down_proj.weight")


def csc(w: torch.Tensor):
    """w (out, in) -> colptr (in + 1), rows, vals: each input's nonzero outputs, ascending."""
    nz = w.t().nonzero()
    cols, rows = nz[:, 0], nz[:, 1]
    colptr = torch.zeros(w.shape[1] + 1, dtype=torch.int64)
    colptr[1:] = torch.bincount(cols, minlength=w.shape[1]).cumsum(0)
    return (colptr.to(torch.int32).numpy(), rows.to(torch.int32).numpy(),
            w[rows, cols].to(torch.float32).numpy())


def embedding_csr(c, lay, n_vocab):
    """The embedding as CSR rows (rowptr, cols, vals): rv-tiny's embedding_matrix, a family at a
    time, never dense."""
    feats = sorted(c.token_fns)
    cols = np.array([lay.dim(d) for d in feats])
    ballast = np.arange(lay.off, lay.off + lay.m)

    def vec(t):
        return np.array([float(c.feature_value(d, t)) for d in feats])

    rows_c, rows_v, lens = [None] * n_vocab, [None] * n_vocab, np.zeros(n_vocab, dtype=np.int64)

    def put(t, r):
        nzc = cols[r != 0]
        cc = np.concatenate([ballast, nzc])
        vv = np.concatenate([np.full(lay.m, lay.ballast), r[r != 0]])
        order = np.argsort(cc)
        rows_c[t], rows_v[t], lens[t] = cc[order], vv[order], len(cc)

    for name, (base, size) in S.FAMILIES.items():
        v0 = vec(base)
        nb = (size - 1).bit_length()
        if size <= 4096:
            block = np.stack([vec(base + i) for i in range(size)])
        else:
            deltas = np.stack([vec(base + (1 << i)) - v0 for i in range(nb)])
            bits = (np.arange(size)[:, None] >> np.arange(nb)[None, :]) & 1
            block = v0[None, :] + bits @ deltas
            for t in (base + size - 1, base + size // 3):
                assert (vec(t) == block[t - base]).all(), (name, t)
        for i in range(size):
            put(base + i, block[i])
    for name, t in S.CONTROLS.items():
        put(t, vec(t))
    for t in range(n_vocab):  # PAD and padding rows: zero
        if rows_c[t] is None:
            rows_c[t], rows_v[t] = np.zeros(0, np.int64), np.zeros(0)
    rowptr = np.zeros(n_vocab + 1, dtype=np.int64)
    rowptr[1:] = np.cumsum(lens)
    cc = np.concatenate(rows_c).astype(np.int32)
    vv = np.concatenate(rows_v).astype(np.float32)
    check_bf16("embedding", torch.tensor(vv))
    return rowptr.astype(np.int32), cc, vv


def main():
    assert S.MAP == "doom", "run with QWENPUTER_MAP=doom"
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    c = core.build()
    lay = pick_layout(len(c.dims))
    n_chip = c.n_layers()
    n_vocab = -(-(S.VOCAB + 1) // 64) * 64
    tc = dict(BASE, hidden_size=lay.hidden, vocab_size=n_vocab, num_hidden_layers=n_chip + 1,
              layer_types=["linear_attention"] * n_chip + ["full_attention"],
              linear_num_key_heads=c.target.key_heads, linear_num_value_heads=c.target.value_heads,
              intermediate_size=c.target.intermediate)
    print(f"circuit {c.stats()}, hidden {lay.hidden}, built in {time.time() - t0:.0f}s", flush=True)

    pack, nnz = {}, 0

    def sink(L, layer):
        nonlocal nnz
        for k in MATS:
            cp, r, v = csc(layer[k])
            pack[f"{L}.{k}.colptr"], pack[f"{L}.{k}.rows"], pack[f"{L}.{k}.vals"] = cp, r, v
            nnz += len(v)
        pack[f"{L}.conv"] = layer["linear_attn.conv1d.weight"][:, 0, :].numpy().astype(np.float32)
        for k in ("linear_attn.A_log", "linear_attn.dt_bias", "linear_attn.norm.weight",
                  "input_layernorm.weight", "post_attention_layernorm.weight"):
            pack[f"{L}.{k}"] = layer[k].numpy().astype(np.float32)
        if L % 25 == 0:
            print(f"  layer {L}/{n_chip}: {nnz:,} nonzero weights so far, "
                  f"{time.time() - t0:.0f}s", flush=True)

    emit_layers(c, lay, tc, 0, "model.", None, set(c.keep), sink=sink)
    pack["emb.rowptr"], pack["emb.cols"], pack["emb.vals"] = embedding_csr(c, lay, n_vocab)
    norm = np.full(lay.hidden, -1.0, dtype=np.float32)  # rv-tiny's: only the code survives
    for d in c.keep:
        norm[lay.dim(d)] = CODE_GAIN
    pack["norm"] = norm
    meta = {"map": S.MAP, "layers": n_chip, "hidden": lay.hidden, "ballast_dims": lay.m,
            "ballast": lay.ballast, "vocab": n_vocab, "chip_vocab": S.VOCAB, "pad": S.VOCAB,
            "key_heads": c.target.key_heads, "value_heads": c.target.value_heads,
            "intermediate": c.target.intermediate, "conv": tc["linear_conv_kernel_dim"],
            "code_dims": sorted(lay.dim(d) for d in c.keep), "circuit": c.stats(),
            "nonzero_weights": nnz}
    pack["meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez(OUT / "pack.npz", **pack)
    size = (OUT / "pack.npz").stat().st_size
    print(f"{OUT / 'pack.npz'}: {n_chip} layers, {nnz:,} nonzero weights, embedding "
          f"{len(pack['emb.vals']):,} entries, {size / 1e6:.0f} MB, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
