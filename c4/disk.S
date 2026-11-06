// The ROM disk's files, linked into the DROM. The paths come from build.sh.
    .section .rodata.disk, "a"
    .globl qd_c4, qd_c4_end, qd_hello, qd_hello_end, qd_fib, qd_fib_end, qd_mandel, qd_mandel_end
qd_c4:    .incbin C4_SRC
qd_c4_end:
qd_hello: .incbin HELLO_SRC
qd_hello_end:
qd_fib:   .incbin FIB_SRC
qd_fib_end:
qd_mandel: .incbin MANDEL_SRC
qd_mandel_end:
