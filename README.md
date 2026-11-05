# qwendoom

The whole of Doom, bare metal, inside Qwen.

[Qwenputer](https://github.com/divagr18/qwenputer) compiles a RISC-V CPU (RV32IM + Zba) into the
weights of a Qwen3.5 model: every token the model generates is a step of the machine, and its RAM
lives in the Gated DeltaNet layers' recurrent state. Stock Qwenputer has 32 KB of RAM. Doom needs
megabytes.

This repo grows the machine to Doom's size and runs Doom on it:

- no OS, no host disk, no paging: Doom ([doomgeneric](https://github.com/ozkl/doomgeneric))
  compiled to RV32 with a small libc, its first instruction Doom's own startup code;
- the code (253 KB) in the model's ROM, the game data (a 5.5 MB Freedoom WAD) in its read-only
  memory, the heap, stack and framebuffer in its 4 MB of RAM: all of it in the model's state,
  loaded once at the start;
- one continuous run, every token the model's own greedy prediction, each one checked against
  Qwenputer's reference CPU as it comes;
- frames read straight out of the model's memory (the screen buffer and palette in its RAM heads).

## Status

Doom boots on the model, token for token equal to the reference machine:

```
                           Doom Generic 0.1
Z_Init: Init zone memory allocation daemon.
zone memory: 49290, 300000 allocated for zone
Using . for configuration and saves
V_Init: allocate screens.
M_LoadDefaults: Load system defaults.
```

No frame yet. The boot is ~63M tokens and a gameplay frame ~2.5M, run on a MacBook CPU (M4) at
~800 tokens/s: a frame every ~50 minutes, the first one about a day in.

## The memory mod

`qwenputer/` is Qwenputer's `experimental/riscv` at `bec1a0c` with a second memory map,
`QWENPUTER_MAP=doom` (the default map is unchanged and passes Qwenputer's tests):

|                 | stock (P2) | doom                       |
|-----------------|------------|----------------------------|
| RAM             | 32 KB      | 4 MB at `0x0`              |
| read-only data  | 128 KB     | 8 MB at `0x1000000`        |
| ROM (code)      | 256 KB     | 256 KB at `0x2000000`      |
| layers          | 11         | 328                        |
| attention heads | 99         | 5,171                      |

One RAM word per DeltaNet state row, 16 KB per layer: the compiler's placer spills the extra
memory heads into new layers. `core.py`'s address decode is generalized to the map's sizes.

## Running it fast enough: the stepper

A dense 328-layer Qwen3.5 would be tens of billions of numbers. `model/pack.py` builds the model a
layer at a time and keeps only its nonzero weights (43.9M, 520 MB). `model/stepper.c` runs
Qwen3.5's decode step on them, with the same bf16 rounding points as transformers, and is checked
token for token against the reference machine.

| step | tokens/s |
|------|---------:|
| skip zero weights and zero inputs, SIMD runs of rows | 79 |
| ballast sums cached, `silu(x < -89) = -0` | 132 |
| 4 threads per layer, each product reads only its inputs, bf16 weights, conv reads only its taps | 265 |
| skip address rows that provably can't match | 797 |

The last step is where the time was. On every token, every memory layer compares the address
against all of its rows, and all but one row lose. The stepper proves this before computing:
for each attention head's 128 query (or key) channels it bounds the largest pre-activation from
the inputs, with the bias column folded into the address-bit columns so the bound is the exact
maximum over the rows, plus a bound on float rounding. A head whose bound is below -100 outputs
exactly -0 after the SiLU, so it is skipped. Values are computed only for a head that can write,
from the last 3 tokens' inputs (kept, and checkpointed). Per token, 93 of 5,248 query heads are
left to compute.

## Layout

- `port/`: the platform layer (video, clock, the WAD as a ROM disk), the libc, the linker map,
  `wad.py` (cuts the IWAD to the demo's map, every lump in place), `build.sh`.
- `tools/rvemu.c`: a fast C emulator of the same machine (the whole demo, 7,155 frames, 11.5G
  instructions, in 21 s; frames pixel-identical to native Doom). `-touch` records every word a run
  reads or writes. `tools/refcheck.py`: the reference machine at instruction N, to compare.
- `model/pack.py`: the sparse model. `model/stepper.c`, `model/stepper.py`: the stepper.
  `model/run.py`: the long run, every token checked, checkpoints every 15 minutes (resumes),
  frames to `model/build/run/frames/`.

## Reproduce

Needs a Mac or Linux box, `uv`, Python 3.12+ with numpy, and a C compiler.

```bash
curl -LO https://github.com/freedoom/freedoom/releases/download/v0.13.0/freedoom-0.13.0.zip
unzip -j freedoom-0.13.0.zip freedoom-0.13.0/freedoom1.wad
git clone https://github.com/ozkl/doomgeneric port/build/doomgeneric   # (the WAD cut reads its tables)
git -C port/build/doomgeneric checkout e257a09
python port/wad.py freedoom1.wad port/build/doom.wad
port/build.sh                      # port/build/doom.elf
mkdir -p tools/build && cc -O2 tools/rvemu.c -o tools/build/rvemu
tools/build/rvemu port/build/doom.elf -q     # the whole demo, in seconds
QWENPUTER_MAP=doom PYTHONPATH=qwenputer python model/pack.py
QWENPUTER_MAP=doom PYTHONPATH=qwenputer python model/stepper.py port/build/doom.elf --tokens 8000 --check
QWENPUTER_MAP=doom PYTHONPATH=qwenputer python model/run.py port/build/doom.elf
```

## Credits and licenses

- Qwenputer by Divyansh Agrawal ([divagr18/qwenputer](https://github.com/divagr18/qwenputer)), MIT:
  `qwenputer/LICENSE`.
- doomgeneric by ozkl ([ozkl/doomgeneric](https://github.com/ozkl/doomgeneric)), GPL-2.0, fetched
  at a pinned commit by `port/build.sh`, not included here.
- Freedoom ([freedoom.github.io](https://freedoom.github.io)), BSD, downloaded, not included.
- Everything else here (port/, tools/, model/) is GPL-2.0, as the port links doomgeneric: `LICENSE`.
