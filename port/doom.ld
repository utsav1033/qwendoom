/* The Doom machine's address map: Qwenputer's (env/rom.ld) with room for Doom.
   RAM 4 MB at 0 (data / TLS / bss / heap from 0x100, the stack at the top), read-only data in the
   DROM at 0x0100_0000 (8 MB: .rodata and the game data), code in the ROM at 0x0200_0000 (256 KB,
   as Qwenputer's: a jump names a 16-bit word index). crt0 points tp at __tls_base. */
ENTRY(_start)
MEMORY
{
  RAM (rw) : ORIGIN = 0x100, LENGTH = 0x400000 - 0x100
  DROM (r) : ORIGIN = 0x1000000, LENGTH = 8M
  ROM (rx) : ORIGIN = 0x2000000, LENGTH = 256K
}
SECTIONS
{
  .text : { *(.text.init) *(.text .text.*) } > ROM
  .rodata : {
    *(.rodata .rodata.* .srodata .srodata.*)
    *(.init_array .init_array.* .fini_array .fini_array.* .eh_frame .eh_frame_hdr)
  } > DROM
  .data : { *(.data .data.* .sdata .sdata.*) } > RAM
  .tdata : { __tls_base = .; *(.tdata .tdata.*) } > RAM
  .tbss : { *(.tbss .tbss.*) } > RAM
  .bss : {
    . = . + SIZEOF(.tbss);
    *(.bss .bss.* .sbss .sbss.*) *(COMMON)
    . = ALIGN(16);
    __heap_start = .;
  } > RAM
  __heap_end = 0x400000 - 0x10000;  /* 64 KB of stack at the top of RAM */
  _end = __heap_start;
}
