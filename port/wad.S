// The ROM disk's one file: the game data, linked into the DROM. WAD_PATH comes from build.sh.
    .section .rodata.wad, "a"
    .balign 4
    .globl qd_wad, qd_wad_end
qd_wad:
    .incbin WAD_PATH
qd_wad_end:
