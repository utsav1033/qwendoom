"""Run a program on the Doom machine's Qwen3.5 layers (model/stepper.c over model/build/pack.npz).

The run is rv-tiny's (qwenputer/chipformer/rv/firmware.py): the program's snapshot (ROM, DROM,
RAM, registers, PC; core.snapshot) written into the layers' recurrent states, then PAD and GO
fed, then every token the model's own greedy prediction from the one before. With ``check``,
each generated token is compared, as it comes, with the reference machine's trace (ref.Machine,
the oracle): the first difference stops the run.

Run: QWENPUTER_MAP=doom PYTHONPATH=qwenputer python model/stepper.py port/build/doom.elf \
         [--tokens N] [--check]
"""

import argparse
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from chipformer.q35.ir import Head
from chipformer.rv import core, elf, ref
from chipformer.rv import spec as S

HERE = Path(__file__).resolve().parent
MATS = ("linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
        "linear_attn.in_proj_b.weight", "linear_attn.in_proj_a.weight",
        "linear_attn.out_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
        "mlp.down_proj.weight")


def build_lib() -> Path:
    src, lib = HERE / "stepper.c", HERE / "build" / "libstepper.dylib"
    if not lib.exists() or lib.stat().st_mtime < src.stat().st_mtime:
        lib.parent.mkdir(exist_ok=True)
        subprocess.run(["cc", "-O3", "-mcpu=native", "-ffp-contract=off", "-shared", "-fPIC",
                        str(src), "-o", str(lib)], check=True)
    return lib


class Stepper:
    def __init__(self, pack=HERE / "build" / "pack.npz", circuit=None):
        p = np.load(pack)
        self.meta = meta = json.loads(bytes(p["meta"]).decode())
        assert meta["map"] == S.MAP
        self.keep = {k: np.ascontiguousarray(p[k]) for k in p.files if k != "meta"}
        k = self.keep
        lib = self.lib = ctypes.CDLL(str(build_lib()))
        P, I = ctypes.c_void_p, ctypes.c_int
        lib.st_new.restype = P
        lib.st_new.argtypes = [I] * 6
        lib.st_mat.argtypes = [P, I, I, I, I, P, P, P]
        lib.st_layer.argtypes = [P, I] + [P] * 6
        lib.st_state_rows.argtypes = [P, I, I, I, P, P]
        lib.st_reset.argtypes = [P]
        lib.st_embed.argtypes = [P, P, P, P, P, I, P]
        lib.st_step.argtypes = [P, I, I]
        lib.st_run.argtypes = [P, P, I, P, I, ctypes.c_long, P]
        lib.st_nonfinite.argtypes = [P]
        lib.st_save.argtypes = lib.st_load.argtypes = [P, ctypes.c_char_p]
        lib.st_state.restype = P
        lib.st_state.argtypes = [P, I]
        D, F, NK, NV = meta["hidden"], meta["intermediate"], meta["key_heads"], meta["value_heads"]
        self.nl, self.r = meta["layers"], NV // NK
        self.m = lib.st_new(self.nl, D, F, NK, NV, meta["vocab"])
        lib.st_ballast.argtypes = [P, I]
        lib.st_ballast(self.m, meta["ballast_dims"])  # the layout's ballast dims lead (off = 0)
        C = 2 * NK * 128 + NV * 128
        dims = {MATS[0]: (D, C), MATS[1]: (D, NV * 128), MATS[2]: (D, NV), MATS[3]: (D, NV),
                MATS[4]: (NV * 128, D), MATS[5]: (D, F), MATS[6]: (D, F), MATS[7]: (F, D)}
        ptr = lambda a: a.ctypes.data
        for L in range(self.nl):
            for j, mat in enumerate(MATS):
                nin, nout = dims[mat]
                # the stepper keeps the weights as bf16 (they are bf16 in the model)
                assert not (k[f"{L}.{mat}.vals"].view(np.uint32) & 0xFFFF).any(), (L, mat)
                lib.st_mat(self.m, L, j, nin, nout, ptr(k[f"{L}.{mat}.colptr"]),
                           ptr(k[f"{L}.{mat}.rows"]), ptr(k[f"{L}.{mat}.vals"]))
            lib.st_layer(self.m, L, ptr(k[f"{L}.conv"]), ptr(k[f"{L}.linear_attn.A_log"]),
                         ptr(k[f"{L}.linear_attn.dt_bias"]), ptr(k[f"{L}.linear_attn.norm.weight"]),
                         ptr(k[f"{L}.input_layernorm.weight"]),
                         ptr(k[f"{L}.post_attention_layernorm.weight"]))
        code = np.array(meta["code_dims"], dtype=np.int32)
        self.keep["code"] = code
        lib.st_embed(self.m, ptr(k["emb.rowptr"]), ptr(k["emb.cols"]), ptr(k["emb.vals"]),
                     ptr(k["norm"]), len(code), ptr(code))
        self.c = circuit or core.build()
        assert self.c.n_layers() == self.nl
        heads = {}
        for L in range(self.nl):  # hf.head_slots, in one pass
            heads.setdefault(L, [])
        for i, n in enumerate(self.c.nodes):
            if isinstance(n, Head):
                heads[(n.level - 1) // 2].append(i)
        self.slots = {i: (L, j) for L, hs in heads.items() for j, i in enumerate(hs)}

    def load(self, machine):
        """The machine's snapshot into the recurrent states (as hf.inject does)."""
        self.lib.st_reset(self.m)
        core.snapshot(self.c, machine)
        for ni, (L, j) in self.slots.items():
            init = self.c.nodes[ni].init or []
            for x, rows in enumerate(init):
                if not rows:
                    continue
                keys = np.array(sorted(rows), dtype=np.int32)
                vals = np.ascontiguousarray(np.stack([rows[r] for r in keys]), dtype=np.float32)
                self.lib.st_state_rows(self.m, L, j * self.r + x, len(keys), keys.ctypes.data,
                                       vals.ctypes.data)

    def save(self, path):
        assert self.lib.st_save(self.m, str(path).encode()) == 0, path

    def restore(self, path):
        assert self.lib.st_load(self.m, str(path).encode()) == 0, path

    def state(self, layer):
        """Layer's recurrent state, (value heads, K, K), a view of the stepper's own."""
        nv = self.meta["value_heads"]
        buf = (ctypes.c_float * (nv * 128 * 128)).from_address(self.lib.st_state(self.m, layer))
        return np.ctypeslib.as_array(buf).reshape(nv, 128, 128)

    def start(self):
        """Feed PAD and GO; returns the first generated token."""
        self.lib.st_step(self.m, S.VOCAB, 0)
        return self.lib.st_step(self.m, S.tok("GO"), 1)

    def step(self, t):
        return self.lib.st_step(self.m, t, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("elf")
    ap.add_argument("--tokens", type=int, default=10000)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    st = Stepper()
    prog = elf.program(args.elf)
    st.load(ref.Machine.from_program(prog))
    print(f"loaded: {st.nl} layers, snapshot in, {time.time() - t0:.0f}s", flush=True)
    oracle, want = ref.Machine.from_program(prog), []
    t1, t, n, out = time.time(), st.start(), 0, bytearray()
    while n < args.tokens:
        if args.check:
            while not want:
                want = oracle.record()
            w = want.pop(0)
            if t != w:
                print(f"DIVERGED at token {n}: model {S.name(t) if t < S.VOCAB else t}, "
                      f"reference {S.name(w)} (after {oracle.steps} instructions)")
                return 1
        n += 1
        if t < S.VOCAB and S.family(t)[0] == "BYTE":
            out.append(S.family(t)[1] & 255)
        if t == S.tok("HALT"):
            break
        t = st.step(t)
        if n % 1000 == 0:
            dt = time.time() - t1
            print(f"  {n} tokens, {n / dt:.0f} tokens/s"
                  + (f", {oracle.steps} instructions, matching" if args.check else ""), flush=True)
    dt = time.time() - t1
    print(f"{n} tokens in {dt:.1f}s ({n / dt:.0f} tokens/s)"
          + (", every one equal to the reference" if args.check else ""))
    print("output:", bytes(out)[:200])
    assert st.lib.st_nonfinite(st.m) == 0, "a non-finite conv input (the zero-tap shortcut is off)"


if __name__ == "__main__":
    sys.exit(main())
