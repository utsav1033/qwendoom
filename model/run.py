"""The long run: Doom on the Doom machine's layers, from the snapshot on, every token checked.

As model/stepper.py's run (the snapshot into the states, PAD and GO, then the model's own greedy
tokens), with each token compared to the reference machine's trace as it comes, and:

- a checkpoint every --every seconds: the stepper's states and the run's own state (the token
  to feed next, the reference machine, the output so far) into one of two slots, then a pointer
  to it, so a crash leaves the last whole one; a new start resumes from it;
- at every frame (the putchar 12 port/i_video_qwen.c's I_FinishUpdate sends) the screen read
  out of the model's own memory, the RAM heads' recurrent states (a word's bit i is A_BIT at
  column 2 + i, where SW and the snapshot write it, or at 34 + i, where an SB/SH merge does: a
  load reads the two added), to frames/NNNN.png with Doom's palette (and
  its gamma table 0, which doomgeneric applies on the host side), and checked against the
  reference machine's RAM.

Run: QWENPUTER_MAP=doom PYTHONPATH=qwenputer python model/run.py port/build/doom.elf \
         [--dir model/build/run] [--every 900]
"""

import argparse
import json
import os
import pickle
import re
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np

from chipformer.q35.ir import A_BIT
from chipformer.rv import elf, ref
from chipformer.rv import spec as S

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stepper import Stepper  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FRAME_MARK = 12
W, H = 320, 200


def symbols(path):
    """An ELF32's symbol table: name -> address."""
    b = Path(path).read_bytes()
    shoff, = struct.unpack_from("<I", b, 0x20)
    shentsize, shnum = struct.unpack_from("<HH", b, 0x2E)
    sh = [struct.unpack_from("<10I", b, shoff + i * shentsize) for i in range(shnum)]
    out = {}
    for s in sh:
        if s[1] != 2:  # SHT_SYMTAB
            continue
        strtab = sh[s[6]]
        for o in range(s[4], s[4] + s[5], 16):
            name, value = struct.unpack_from("<II", b, o)
            end = b.index(b"\0", strtab[4] + name)
            out[b[strtab[4] + name:end].decode()] = value
    return out


def gamma0():
    """Doom's gamma table 0 (tables.c), which doomgeneric applies to the palette."""
    src = (ROOT / "port/build/doomgeneric/doomgeneric/tables.c").read_text()
    body = src[src.index("gammatable[5][256]"):]
    return np.array([int(v) for v in re.findall(r"\d+", body[body.index("{"):])[:256]], np.uint8)


def png(path, rgb):
    raw = b"".join(b"\0" + rgb[y].tobytes() for y in range(rgb.shape[0]))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", rgb.shape[1], rgb.shape[0], 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


class Memory:
    """RAM words read out of the model's recurrent states."""

    def __init__(self, st):
        self.st = st
        index = {id(n): i for i, n in enumerate(st.c.nodes)}
        self.where = [st.slots[index[id(st.c.tables[f"ram{h}"])]] for h in range((S.RAM_WORDS >> 12) << 4)]

    def words(self, addr, n):
        wi = (addr - S.RAM_BASE) // 4 + np.arange(n)
        head, x, key = (wi & 15) | (wi >> 12) << 4, wi >> 4 & 1, wi >> 5 & 127
        out = np.zeros(n, np.uint32)
        for h in np.unique(head):
            sel = head == h
            L, j = self.where[h]
            rows = self.st.state(L)[j * self.st.r + x[sel], key[sel]]
            bits = (rows[:, 2:34] + rows[:, 34:66] > A_BIT / 2).astype(np.uint32)
            out[sel] = (bits << np.arange(32, dtype=np.uint32)).sum(1, dtype=np.uint32)
        return out

    def bytes(self, addr, n):
        a0 = addr & ~3
        w = self.words(a0, (addr + n - a0 + 3) // 4)
        return w.view(np.uint8)[addr - a0:addr - a0 + n]


def oracle_bytes(m, addr, n):
    a0 = addr & ~3
    w = np.array(m.ram[(a0 - S.RAM_BASE) // 4:][: (addr + n - a0 + 3) // 4], np.uint32)
    return w.view(np.uint8)[addr - a0:addr - a0 + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("elf")
    ap.add_argument("--dir", default=str(ROOT / "model/build/run"))
    ap.add_argument("--every", type=float, default=900)
    ap.add_argument("--tokens", type=int, default=0, help="stop after this many tokens (0: run on)")
    args = ap.parse_args()
    d = Path(args.dir)
    (d / "frames").mkdir(parents=True, exist_ok=True)
    log = open(d / "progress.log", "a")

    def say(msg):
        line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    t0 = time.time()
    st = Stepper()
    mem = Memory(st)
    sym = symbols(args.elf)
    vbuf, pal, gamma = sym["I_VideoBuffer"], sym["qd_palette"], gamma0()
    latest = d / "latest.json"
    if latest.exists():
        slot = json.loads(latest.read_text())["slot"]
        st.restore(d / f"ckpt{slot}.state")
        run = pickle.loads((d / f"ckpt{slot}.pkl").read_bytes())
        say(f"resumed from slot {slot}: token {run['n']:,}, {run['oracle'].steps:,} instructions, "
            f"{run['frames']} frames ({time.time() - t0:.0f}s)")
    else:
        prog = elf.program(args.elf)
        st.load(ref.Machine.from_program(prog))
        run = dict(n=0, t=st.start(), out=bytearray(), oracle=ref.Machine.from_program(prog),
                   want=[], frames=0, seconds=0.0, gen=0)
        say(f"started: {st.nl} layers, snapshot in ({time.time() - t0:.0f}s)")

    def checkpoint():
        slot = run["gen"] % 2
        run["gen"] += 1  # (the next one goes to the other slot, after a resume too)
        st.save(d / f"ckpt{slot}.state")
        (d / f"ckpt{slot}.pkl").write_bytes(pickle.dumps(run))
        tmp = d / "latest.json.tmp"
        tmp.write_text(json.dumps({"slot": slot, "n": run["n"], "steps": run["oracle"].steps}))
        os.replace(tmp, latest)

    def frame():
        k = run["frames"]
        screen = int(mem.words(vbuf, 1)[0])
        pix, palette = mem.bytes(screen, W * H), mem.bytes(pal, 768)
        same = (np.array_equal(pix, oracle_bytes(run["oracle"], screen, W * H))
                and np.array_equal(palette, oracle_bytes(run["oracle"], pal, 768)))
        rgb = gamma[palette.reshape(256, 3)][pix].reshape(H, W, 3)
        png(d / "frames" / f"{k:04d}.png", rgb)
        say(f"FRAME {k} at token {run['n']:,} ({run['oracle'].steps:,} instructions): screen read "
            f"from the model's memory, {'equal to' if same else 'DIFFERENT from'} the reference's")
        run["frames"] += 1
        return same

    oracle, want, out = run["oracle"], run["want"], run["out"]
    t, n = run["t"], run["n"]
    last_ck, last_log, n_log, t_run = time.time(), time.time(), n, time.time()
    while not args.tokens or n < args.tokens:
        while not want:
            want.extend(oracle.record())
        w = want.pop(0)
        if t != w:
            say(f"DIVERGED at token {n:,}: model {S.name(t) if t < S.VOCAB else t}, reference "
                f"{S.name(w)} (after {oracle.steps:,} instructions)")
            return 1
        n += 1
        if S.family(t)[0] == "BYTE":
            b = S.family(t)[1] & 255
            out.append(b)
            if b == FRAME_MARK:
                run.update(t=st.step(t), n=n)
                t = run["t"]
                frame()
                continue
            if b == 10:
                say("output: " + out[out.rfind(b"\n", 0, len(out) - 1) + 1:-1].decode("latin-1"))
        if t == S.tok("HALT"):
            say("HALT")
            break
        t = st.step(t)
        now = time.time()
        if now - last_log >= 60:
            rate = (n - n_log) / (now - last_log)
            say(f"token {n:,}, {oracle.steps:,} instructions, {rate:.0f} tokens/s, all equal")
            last_log, n_log = now, n
        if now - last_ck >= args.every:
            run.update(t=t, n=n, seconds=run["seconds"] + now - t_run)
            t_run = now
            checkpoint()
            last_ck = time.time()
    run.update(t=t, n=n, seconds=run["seconds"] + time.time() - t_run)
    checkpoint()
    say(f"stopped at token {n:,} ({oracle.steps:,} instructions), checkpoint saved")


if __name__ == "__main__":
    sys.exit(main())
