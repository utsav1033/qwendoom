// The C library Doom needs, for the Qwenputer machine: no OS, no atomics, no floats in hardware.
// Output goes to putchar (ecall a7 = 1), exit to ecall a7 = 93. Files are a ROM disk: one game
// data file, linked into the DROM (wad.S); every other fopen fails. malloc is a bump allocator
// (Doom takes its zone once and frees little), so every run allocates the same addresses.
#include <ctype.h>
#include <errno.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "rt.h"

// --- memory ----------------------------------------------------------------------------------

extern char __heap_start[], __heap_end[];
static char *brk_ = __heap_start;

void *malloc(size_t n) {
    n = (n + 15) & ~(size_t)15;
    if (n + 16 > (size_t)(__heap_end - brk_)) return NULL;
    *(size_t *)brk_ = n;  // the size, for realloc
    void *p = brk_ + 16;
    brk_ += n + 16;
    return p;
}
void free(void *p) { (void)p; }
void *calloc(size_t a, size_t b) {
    void *p = malloc(a * b);
    if (p) memset(p, 0, a * b);
    return p;
}
void *realloc(void *p, size_t n) {
    if (!p) return malloc(n);
    size_t old = *(size_t *)((char *)p - 16);
    if (n <= old) return p;
    void *q = malloc(n);
    if (q) memcpy(q, p, old);
    return q;
}

// --- strings ---------------------------------------------------------------------------------

void *memcpy(void *d, const void *s, size_t n) {
    unsigned char *a = d;
    const unsigned char *b = s;
    if (!(((uintptr_t)a | (uintptr_t)b | n) & 3)) {
        uint32_t *aw = d;
        const uint32_t *bw = s;
        for (n >>= 2; n--;) *aw++ = *bw++;
        return d;
    }
    while (n--) *a++ = *b++;
    return d;
}
void *memmove(void *d, const void *s, size_t n) {
    unsigned char *a = d;
    const unsigned char *b = s;
    if (a < b) return memcpy(d, s, n);
    while (n--) a[n] = b[n];
    return d;
}
void *memset(void *d, int c, size_t n) {
    unsigned char *a = d;
    if (!(((uintptr_t)a | n) & 3)) {
        uint32_t w = (unsigned char)c * 0x01010101u, *aw = d;
        for (n >>= 2; n--;) *aw++ = w;
        return d;
    }
    while (n--) *a++ = (unsigned char)c;
    return d;
}
int memcmp(const void *x, const void *y, size_t n) {
    const unsigned char *a = x, *b = y;
    for (; n--; a++, b++)
        if (*a != *b) return *a - *b;
    return 0;
}
void *memchr(const void *s, int c, size_t n) {
    const unsigned char *a = s;
    for (; n--; a++)
        if (*a == (unsigned char)c) return (void *)a;
    return NULL;
}
size_t strlen(const char *s) {
    const char *p = s;
    while (*p) p++;
    return p - s;
}
int strcmp(const char *a, const char *b) {
    while (*a && *a == *b) a++, b++;
    return (unsigned char)*a - (unsigned char)*b;
}
int strncmp(const char *a, const char *b, size_t n) {
    for (; n; n--, a++, b++)
        if (*a != *b || !*a) return (unsigned char)*a - (unsigned char)*b;
    return 0;
}
int strcasecmp(const char *a, const char *b) {
    while (*a && tolower((unsigned char)*a) == tolower((unsigned char)*b)) a++, b++;
    return tolower((unsigned char)*a) - tolower((unsigned char)*b);
}
int strncasecmp(const char *a, const char *b, size_t n) {
    for (; n; n--, a++, b++) {
        int x = tolower((unsigned char)*a), y = tolower((unsigned char)*b);
        if (x != y || !x) return x - y;
    }
    return 0;
}
char *strcpy(char *d, const char *s) {
    char *r = d;
    while ((*d++ = *s++)) {}
    return r;
}
char *strncpy(char *d, const char *s, size_t n) {
    size_t i = 0;
    for (; i < n && s[i]; i++) d[i] = s[i];
    for (; i < n; i++) d[i] = 0;
    return d;
}
char *strcat(char *d, const char *s) {
    strcpy(d + strlen(d), s);
    return d;
}
char *strncat(char *d, const char *s, size_t n) {
    char *e = d + strlen(d);
    while (n-- && *s) *e++ = *s++;
    *e = 0;
    return d;
}
char *strchr(const char *s, int c) {
    for (;; s++) {
        if (*s == (char)c) return (char *)s;
        if (!*s) return NULL;
    }
}
char *strrchr(const char *s, int c) {
    const char *r = NULL;
    for (;; s++) {
        if (*s == (char)c) r = s;
        if (!*s) return (char *)r;
    }
}
char *strstr(const char *h, const char *n) {
    size_t k = strlen(n);
    for (; *h; h++)
        if (!strncmp(h, n, k)) return (char *)h;
    return k ? NULL : (char *)h;
}
char *strdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *d = malloc(n);
    return d ? memcpy(d, s, n) : NULL;
}
int (toupper)(int c) { return c >= 'a' && c <= 'z' ? c - 32 : c; }
int (tolower)(int c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; }
int (isspace)(int c) { return c == ' ' || (c >= '\t' && c <= '\r'); }
int (isdigit)(int c) { return c >= '0' && c <= '9'; }
int (isalpha)(int c) { return (c | 32) >= 'a' && (c | 32) <= 'z'; }
int (isalnum)(int c) { return isalpha(c) || isdigit(c); }
int (isupper)(int c) { return c >= 'A' && c <= 'Z'; }
int (islower)(int c) { return c >= 'a' && c <= 'z'; }
int (isprint)(int c) { return c >= 32 && c < 127; }
int (isxdigit)(int c) { return isdigit(c) || ((c | 32) >= 'a' && (c | 32) <= 'f'); }

long strtol(const char *s, char **end, int base) {
    while (isspace((unsigned char)*s)) s++;
    int neg = *s == '-';
    if (*s == '-' || *s == '+') s++;
    if ((base == 0 || base == 16) && s[0] == '0' && (s[1] | 32) == 'x') s += 2, base = 16;
    else if (base == 0) base = *s == '0' ? 8 : 10;
    long v = 0;
    for (;; s++) {
        int d = isdigit((unsigned char)*s) ? *s - '0'
                : isalpha((unsigned char)*s) ? (*s | 32) - 'a' + 10 : 99;
        if (d >= base) break;
        v = v * base + d;
    }
    if (end) *end = (char *)s;
    return neg ? -v : v;
}
unsigned long strtoul(const char *s, char **end, int base) {
    return (unsigned long)strtol(s, end, base);
}
int atoi(const char *s) { return (int)strtol(s, NULL, 10); }
long atol(const char *s) { return strtol(s, NULL, 10); }
double atof(const char *s) {  // only a configuration file would need this; there is none
    return (double)strtol(s, NULL, 10);
}
int abs(int x) { return x < 0 ? -x : x; }
long labs(long x) { return x < 0 ? -x : x; }

static int errno_;
int *__errno_location(void) { return &errno_; }
char *getenv(const char *n) { (void)n; return NULL; }
int system(const char *c) { (void)c; return -1; }

static void (*atexit_fns[32])(void);
static int n_atexit;
int atexit(void (*f)(void)) {
    if (n_atexit == 32) return -1;
    atexit_fns[n_atexit++] = f;
    return 0;
}
void exit(int code) {
    while (n_atexit) atexit_fns[--n_atexit]();
    for (;;) rv_ecall2(93, code, 0, 0);
}
void abort(void) { exit(134); }

static void swap(char *a, char *b, size_t n) {
    while (n--) {
        char t = *a;
        *a++ = *b, *b++ = t;
    }
}
void qsort(void *base, size_t n, size_t sz, int (*cmp)(const void *, const void *)) {
    char *b = base;  // insertion sort: Doom sorts little
    for (size_t i = 1; i < n; i++)
        for (size_t j = i; j > 0 && cmp(b + (j - 1) * sz, b + j * sz) > 0; j--)
            swap(b + (j - 1) * sz, b + j * sz, sz);
}

// --- formatted output ------------------------------------------------------------------------

typedef struct {
    char *buf;
    size_t cap, len;  // cap 0: to putchar
} Out;

static void out_c(Out *o, char c) {
    if (!o->buf) rv_putchar(c);
    else if (o->len + 1 < o->cap) o->buf[o->len] = c;
    o->len++;
}

static void out_num(Out *o, unsigned long long v, int base, int upper, int neg, int width,
                    int zero, int left, int prec) {
    char t[24];
    int n = 0;
    const char *dig = upper ? "0123456789ABCDEF" : "0123456789abcdef";
    do t[n++] = dig[v % base], v /= base; while (v);
    while (n < prec) t[n++] = '0';
    int len = n + neg;
    if (!left && !zero)
        for (; width > len; width--) out_c(o, ' ');
    if (neg) out_c(o, '-');
    if (!left && zero)
        for (; width > len; width--) out_c(o, '0');
    while (n) out_c(o, t[--n]);
    for (; width > len; width--) out_c(o, ' ');
}

static int format(Out *o, const char *f, va_list ap) {
    for (; *f; f++) {
        if (*f != '%') {
            out_c(o, *f);
            continue;
        }
        int left = 0, zero = 0, width = 0, prec = -1, lng = 0;
        for (f++;; f++) {
            if (*f == '-') left = 1;
            else if (*f == '0') zero = 1;
            else if (*f != '+' && *f != ' ' && *f != '#') break;
        }
        if (*f == '*') width = va_arg(ap, int), f++;
        else while (isdigit((unsigned char)*f)) width = width * 10 + *f++ - '0';
        if (*f == '.') {
            prec = 0, f++;
            if (*f == '*') prec = va_arg(ap, int), f++;
            else while (isdigit((unsigned char)*f)) prec = prec * 10 + *f++ - '0';
        }
        for (; *f == 'l' || *f == 'h' || *f == 'z'; f++) lng += *f == 'l';
        switch (*f) {
        case 'd': case 'i': {
            long long v = lng > 1 ? va_arg(ap, long long) : lng ? va_arg(ap, long) : va_arg(ap, int);
            out_num(o, v < 0 ? -(unsigned long long)v : v, 10, 0, v < 0, width, zero, left, prec);
            break;
        }
        case 'u': case 'x': case 'X': case 'o': {
            unsigned long long v = lng > 1 ? va_arg(ap, unsigned long long)
                                   : lng ? va_arg(ap, unsigned long) : va_arg(ap, unsigned);
            out_num(o, v, *f == 'u' ? 10 : *f == 'o' ? 8 : 16, *f == 'X', 0, width, zero, left,
                    prec);
            break;
        }
        case 'p':
            out_num(o, (uintptr_t)va_arg(ap, void *), 16, 0, 0, width, zero, left, prec);
            break;
        case 'c':
            out_c(o, (char)va_arg(ap, int));
            break;
        case 's': {
            const char *s = va_arg(ap, const char *);
            if (!s) s = "(null)";
            int n = 0;
            while (s[n] && (prec < 0 || n < prec)) n++;
            if (!left) for (; width > n; width--) out_c(o, ' ');
            for (int i = 0; i < n; i++) out_c(o, s[i]);
            for (; width > n; width--) out_c(o, ' ');
            break;
        }
        case 'f': case 'g': case 'e': {  // no floats are printed on a run; keep the integer part
            double v = va_arg(ap, double);
            long long iv = (long long)v;
            out_num(o, iv < 0 ? -iv : iv, 10, 0, iv < 0, width, zero, left, -1);
            break;
        }
        case '%':
            out_c(o, '%');
            break;
        default:
            out_c(o, '%'), out_c(o, *f);
        }
    }
    if (o->buf && o->cap) o->buf[o->len < o->cap ? o->len : o->cap - 1] = 0;
    return (int)o->len;
}

int vsnprintf(char *b, size_t n, const char *f, va_list ap) {
    Out o = {b, n, 0};
    static char dummy;
    if (!b) o.buf = &dummy, o.cap = 0;
    return format(&o, f, ap);
}
int snprintf(char *b, size_t n, const char *f, ...) {
    va_list ap;
    va_start(ap, f);
    int r = vsnprintf(b, n, f, ap);
    va_end(ap);
    return r;
}
int vsprintf(char *b, const char *f, va_list ap) { return vsnprintf(b, (size_t)-1 >> 1, f, ap); }
int sprintf(char *b, const char *f, ...) {
    va_list ap;
    va_start(ap, f);
    int r = vsprintf(b, f, ap);
    va_end(ap);
    return r;
}
int vprintf(const char *f, va_list ap) {
    Out o = {NULL, 0, 0};
    return format(&o, f, ap);
}
int printf(const char *f, ...) {
    va_list ap;
    va_start(ap, f);
    int r = vprintf(f, ap);
    va_end(ap);
    return r;
}
int putchar(int c) {
    rv_putchar((char)c);
    return (unsigned char)c;
}
int puts(const char *s) {
    while (*s) rv_putchar(*s++);
    rv_putchar('\n');
    return 0;
}
int sscanf(const char *s, const char *f, ...) {  // read only from configuration files: none
    (void)s, (void)f;
    return 0;
}

// --- files: the ROM disk ---------------------------------------------------------------------

struct _IO_FILE {
    const unsigned char *data;  // NULL: stdout / stderr
    long size, pos;
};
static struct _IO_FILE std_out, std_err, disk;
FILE *const stdout = &std_out;
FILE *const stderr = &std_err;
FILE *const stdin = &std_out;

extern const unsigned char qd_wad[], qd_wad_end[];
#define WAD_NAME "doom.wad"

FILE *fopen(const char *path, const char *mode) {
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    if (mode[0] != 'r' || strcasecmp(base, WAD_NAME)) return NULL;
    disk.data = qd_wad, disk.size = qd_wad_end - qd_wad, disk.pos = 0;
    return &disk;
}
int fclose(FILE *f) { (void)f; return 0; }
size_t fread(void *p, size_t sz, size_t n, FILE *f) {
    if (!f->data || !sz) return 0;
    long want = (long)(sz * n), left = f->size - f->pos;
    if (want > left) want = left - left % (long)sz;
    memcpy(p, f->data + f->pos, (size_t)want);
    f->pos += want;
    return (size_t)want / sz;
}
size_t fwrite(const void *p, size_t sz, size_t n, FILE *f) {
    if (f->data) return 0;
    const char *c = p;
    for (size_t i = 0; i < sz * n; i++) rv_putchar(c[i]);
    return n;
}
int fseek(FILE *f, long off, int whence) {
    long base = whence == SEEK_SET ? 0 : whence == SEEK_CUR ? f->pos : f->size;
    if (base + off < 0 || base + off > f->size) return -1;
    f->pos = base + off;
    return 0;
}
long ftell(FILE *f) { return f->pos; }
int fflush(FILE *f) { (void)f; return 0; }
int fputc(int c, FILE *f) {
    if (!f->data) rv_putchar((char)c);
    return c;
}
int fputs(const char *s, FILE *f) {
    while (*s) fputc(*s++, f);
    return 0;
}
int vfprintf(FILE *f, const char *fmt, va_list ap) {
    (void)f;
    return vprintf(fmt, ap);
}
int fprintf(FILE *f, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int r = vfprintf(f, fmt, ap);
    va_end(ap);
    return r;
}
int remove(const char *p) { (void)p; return -1; }
int rename(const char *a, const char *b) { (void)a, (void)b; return -1; }
int mkdir(const char *p, unsigned m) { (void)p, (void)m; return -1; }
