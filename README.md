# qwendoom

Real Doom, run by the computer compiled into Qwen's weights.

- `qwenputer/`: Qwenputer's RV32 machine (chipformer, experimental/riscv) by Divyansh Agrawal,
  imported from [divagr18/qwenputer](https://github.com/divagr18/qwenputer) at commit `bec1a0c`
  (MIT, `qwenputer/LICENSE`). This copy gets a larger memory map for Doom.
- `port/`: Doom's platform layer for that machine (video, clock, a ROM disk) and a small libc.
  The engine is [doomgeneric](https://github.com/ozkl/doomgeneric) (GPL-2.0), fetched at a pinned
  commit by `port/build.sh`. The game data is [Freedoom](https://freedoom.github.io) (BSD).

Work in progress.
