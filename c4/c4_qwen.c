// c4's platform for the Qwenputer machine: a ROM disk of source files (disk.S) behind
// open / read / close, and main, which runs c4 on the arguments below. c4 hands open and read
// through to the programs it runs, so a c4 compiled by c4 reads the same disk.
#include <string.h>

typedef struct { const char *name; const char *data, *end; } File;
extern const char qd_c4[], qd_c4_end[], qd_hello[], qd_hello_end[], qd_fib[], qd_fib_end[];
static const File disk[] = {
    {"c4.c", qd_c4, qd_c4_end}, {"hello.c", qd_hello, qd_hello_end}, {"fib.c", qd_fib, qd_fib_end}};
#define NFILES (int)(sizeof disk / sizeof *disk)

static struct { const File *f; int pos; } fds[8];

int open(const char *path, int flags, ...) {
    (void)flags;
    for (int i = 0; i < NFILES; i++)
        if (!strcmp(path, disk[i].name))
            for (int fd = 3; fd < 8; fd++)
                if (!fds[fd].f) return fds[fd].f = &disk[i], fds[fd].pos = 0, fd;
    return -1;
}
long read(int fd, void *p, unsigned long n) {
    if (fd < 3 || fd >= 8 || !fds[fd].f) return -1;
    long left = fds[fd].f->end - fds[fd].f->data - fds[fd].pos;
    if ((long)n > left) n = left;
    memcpy(p, fds[fd].f->data + fds[fd].pos, n);
    fds[fd].pos += n;
    return n;
}
int close(int fd) {
    if (fd < 3 || fd >= 8) return -1;
    fds[fd].f = 0;
    return 0;
}

int c4_main(int argc, char **argv);
#ifndef C4_ARGS
#define C4_ARGS "c4", "hello.c"
#endif
static char *args[] = {C4_ARGS, 0};

int main(void) { return c4_main(sizeof args / sizeof *args - 1, args); }
