#!/bin/bash
# Build c4 (Robert Swierczek's C in four functions) for the Qwenputer machine: c4/build/c4-<name>.elf.
#   c4/build.sh NAME 'ARGS'     e.g. c4/build.sh hello '"c4", "hello.c"'
# Same toolchain and address map as port/build.sh (Doom's), c4's int made 32-bit (pointers are).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=$HERE/../port
OUT=$HERE/build
C4_COMMIT=2feb8c0
ENV=$HERE/../qwenputer/chipformer/rv/env
mkdir -p "$OUT"
if [ ! -d "$OUT/c4" ]; then git clone -q https://github.com/rswier/c4 "$OUT/c4"; fi
git -C "$OUT/c4" checkout -q "$C4_COMMIT"
sed 's/^#define int long long$/#define int int/' "$OUT/c4/c4.c" > "$OUT/c4_32.c"

ZIGPKG=$(uv run --no-project --with ziglang python -I -c \
  "import ziglang, os; print(os.path.dirname(ziglang.__file__))")
ZIG=(uv run --no-project --with ziglang python -I -m ziglang)
INC=$ZIGPKG/lib/libc/include
TARGET=(-target riscv32-freestanding-none -mcpu=generic_rv32+m+zba -mabi=ilp32 -mno-relax)
CFLAGS=("${TARGET[@]}" -O2 -w -nostdlibinc -isystem "$INC/riscv32-linux-musl"
  -isystem "$INC/generic-musl" -isystem "$INC/any-linux-any" -I"$ENV"
  -ffunction-sections -fdata-sections -fno-PIC)

"${ZIG[@]}" cc "${CFLAGS[@]}" -Dmain=c4_main -c "$OUT/c4_32.c" -o "$OUT/c4.o"
"${ZIG[@]}" cc "${CFLAGS[@]}" "-DC4_ARGS=$2" -c "$HERE/c4_qwen.c" -o "$OUT/c4_qwen.o"
"${ZIG[@]}" cc "${CFLAGS[@]}" -c "$PORT/libc.c" -o "$OUT/libc.o"
"${ZIG[@]}" cc "${TARGET[@]}" -c "$ENV/crt0.S" -o "$OUT/crt0.o"
"${ZIG[@]}" cc "${TARGET[@]}" -DC4_SRC="\"$OUT/c4_32.c\"" -DHELLO_SRC="\"$OUT/c4/hello.c\"" \
  -DFIB_SRC="\"$HERE/fib.c\"" -c "$HERE/disk.S" -o "$OUT/disk.o"
# libc.c's ROM disk is Doom's; its wad symbols are given empty here
printf '.section .rodata\n.globl qd_wad, qd_wad_end\nqd_wad:\nqd_wad_end:\n' > "$OUT/nowad.S"
"${ZIG[@]}" cc "${TARGET[@]}" -c "$OUT/nowad.S" -o "$OUT/nowad.o"
"${ZIG[@]}" cc "${TARGET[@]}" -nostartfiles -static -T "$PORT/doom.ld" -Wl,--gc-sections \
  "$OUT/crt0.o" "$OUT/c4.o" "$OUT/c4_qwen.o" "$OUT/libc.o" "$OUT/disk.o" "$OUT/nowad.o" \
  -o "$OUT/c4-$1.elf"
echo "$OUT/c4-$1.elf"
