#!/bin/bash
# Build Doom for the Qwenputer machine (RV32IM + Zba, soft float): port/build/doom.elf.
#   port/build.sh [WAD]     (default port/build/doom.wad, made by port/wad.py)
# Fetches doomgeneric at a pinned commit, compiles its engine with zig's clang against musl's
# headers, and links it with this port (video, platform, libc, ROM disk) and zig's compiler-rt.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/build
WAD=$(cd "$(dirname "${1:-$OUT/doom.wad}")" && pwd)/$(basename "${1:-$OUT/doom.wad}")
DG_COMMIT=e257a09
ENV=$HERE/../qwenputer/chipformer/rv/env
mkdir -p "$OUT/obj"

if [ ! -d "$OUT/doomgeneric" ]; then
  git clone -q https://github.com/ozkl/doomgeneric "$OUT/doomgeneric"
fi
git -C "$OUT/doomgeneric" checkout -q "$DG_COMMIT"
DG=$OUT/doomgeneric/doomgeneric

ZIGPKG=$(uv run --no-project --with ziglang python -I -c \
  "import ziglang, os; print(os.path.dirname(ziglang.__file__))")
ZIG=(uv run --no-project --with ziglang python -I -m ziglang)
INC=$ZIGPKG/lib/libc/include
TARGET=(-target riscv32-freestanding-none -mcpu=generic_rv32+m+zba -mabi=ilp32 -mno-relax)
CFLAGS=("${TARGET[@]}" -O2 -w -nostdlibinc -isystem "$INC/riscv32-linux-musl"
  -isystem "$INC/generic-musl" -isystem "$INC/any-linux-any" -I"$DG" -I"$ENV"
  -DDOOMGENERIC_RESX=320 -DDOOMGENERIC_RESY=200 -ffunction-sections -fdata-sections -fno-PIC)

# the engine: doomgeneric's SRC_DOOM, less its video layer, main and platforms (port/ has its own)
srcs=()
for o in $(sed -n '/^SRC_DOOM/,/^$/p' "$DG/Makefile" | tr ' \\' '\n\n' | grep '\.o$'); do
  case $o in i_video.o|doomgeneric.o|doomgeneric_*.o) continue;; esac
  srcs+=("$DG/${o%.o}.c")
done
srcs+=("$HERE/i_video_qwen.c" "$HERE/dg_qwen.c" "$HERE/libc.c")

objs=()
for c in "${srcs[@]}"; do
  o=$OUT/obj/$(basename "${c%.c}").o
  if [ ! -f "$o" ] || [ "$c" -nt "$o" ]; then "${ZIG[@]}" cc "${CFLAGS[@]}" -c "$c" -o "$o"; fi
  objs+=("$o")
done
"${ZIG[@]}" cc "${TARGET[@]}" -c "$ENV/crt0.S" -o "$OUT/obj/crt0.o"
"${ZIG[@]}" cc "${TARGET[@]}" -DWAD_PATH="\"$WAD\"" -c "$HERE/wad.S" -o "$OUT/obj/wad.o"

"${ZIG[@]}" cc "${TARGET[@]}" -nostartfiles -static -T "$HERE/doom.ld" -Wl,--gc-sections \
  "$OUT/obj/crt0.o" "${objs[@]}" "$OUT/obj/wad.o" -o "$OUT/doom.elf"
echo "$OUT/doom.elf"
