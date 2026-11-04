"""Run a program on Qwenputer's reference machine (ref.Machine, the trace the circuit must write)
for N instructions and print what tools/rvemu prints with -stop N: the PC, the registers and a
hash of RAM. The two machines agree when the lines match.

Run: QWENPUTER_MAP=doom PYTHONPATH=qwenputer python tools/refcheck.py port/build/doom.elf N
"""

import sys
import time

from chipformer.rv import elf, ref
from chipformer.rv import spec as S


def main(path, n):
    n = int(n)
    m = ref.Machine.from_program(elf.program(path))
    t0, toks = time.time(), 0
    while m.steps < n and not m.halted:
        toks += len(m.record())
    dt = time.time() - t0
    h = 1469598103934665603
    for w in m.ram:
        for i in range(4):
            h = ((h ^ (w >> 8 * i & 0xFF)) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    print(f"stop {m.steps} pc {m.pc:08x} ram {h:016x} x " + " ".join(f"{v:x}" for v in m.x[1:]))
    print(f"(reference: {m.steps} instructions, {toks} trace tokens, "
          f"{toks / max(m.steps, 1):.3f} tokens per instruction, {dt:.1f}s; map {S.MAP}; "
          f"output {bytes(m.out)[:60]!r})", file=sys.stderr)


if __name__ == "__main__":
    main(*sys.argv[1:3])
