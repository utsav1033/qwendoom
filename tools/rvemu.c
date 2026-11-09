// A fast emulator of the Doom machine: Qwenputer's RV32IM + Zba at port/doom.ld's address map.
//
// The same machine as the reference (qwenputer/chipformer/rv/ref.py), for checking a program and
// counting its instructions at native speed: loads and stores only to RAM (and loads from the
// DROM), aligned; code only from the ROM; ecall a7 = 1 prints a0, a7 = 93 exits; anything else
// faults and stops. At each FRAME_MARK byte the screen can be saved as a PPM, read from the
// machine's memory as the host will read it from the model's state.
//
//   rvemu doom.elf [-frames N] [-shot K out.ppm] [-vbuf ADDR -pal ADDR] [-q] [-stop N]
//                  [-touch out.bin]
//
// prints the program's output (FRAME_MARK bytes left out), then the instruction count at every
// frame (-q: only the totals). -stop N: after N instructions, the PC, registers and a RAM hash.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RAM_SIZE 0x400000u
#define DROM_BASE 0x1000000u
#define DROM_SIZE 0x800000u
#define ROM_BASE 0x2000000u
#define ROM_SIZE 0x40000u
#define FRAME_MARK 12

static uint8_t ram[RAM_SIZE], drom[DROM_SIZE], rom[ROM_SIZE];
static uint32_t x[32], pc;
static uint64_t icount;

static void fault(const char *why, uint32_t a) {
    fprintf(stderr, "\nfault: %s (address %08x) at pc %08x after %llu instructions\n", why, a, pc,
            (unsigned long long)icount);
    exit(2);
}

// -touch: every RAM and DROM word the program reads or writes, one byte each (RAM's words, then
// the DROM's), written at exit
static const char *touch_path;
static uint8_t ram_t[RAM_SIZE / 4], drom_t[DROM_SIZE / 4];
static void touch_write(void) {
    FILE *f = touch_path ? fopen(touch_path, "wb") : NULL;
    if (!f) return;
    fwrite(ram_t, 1, sizeof ram_t, f), fwrite(drom_t, 1, sizeof drom_t, f), fclose(f);
}

static uint8_t *mem(uint32_t a, int n, int store) {
    if (a & (n - 1)) fault(store ? "misaligned store" : "misaligned load", a);
    if (a + n <= RAM_SIZE) return ram_t[a >> 2] = 1, ram + a;
    if (!store && a >= DROM_BASE && a + n <= DROM_BASE + DROM_SIZE)
        return drom_t[(a - DROM_BASE) >> 2] = 1, drom + (a - DROM_BASE);
    fault(store ? "store out of range" : "load out of range", a);
    return NULL;
}
static uint32_t ld(uint32_t a, int n) {
    uint8_t *p = mem(a, n, 0);
    return n == 1 ? p[0] : n == 2 ? (uint32_t)p[0] | p[1] << 8 : (uint32_t)p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24;
}
static void st(uint32_t a, uint32_t v, int n) {
    uint8_t *p = mem(a, n, 1);
    for (int i = 0; i < n; i++) p[i] = v >> 8 * i;
}

static void load_elf(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) perror(path), exit(1);
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    uint8_t *b = malloc(n);
    fseek(f, 0, SEEK_SET);
    if (fread(b, 1, n, f) != (size_t)n) exit(1);
    fclose(f);
    uint32_t entry, phoff;
    uint16_t phsz, phnum;
    memcpy(&entry, b + 0x18, 4), memcpy(&phoff, b + 0x1c, 4);
    memcpy(&phsz, b + 0x2a, 2), memcpy(&phnum, b + 0x2c, 2);
    for (int i = 0; i < phnum; i++) {
        uint32_t h[8];
        memcpy(h, b + phoff + i * phsz, 32);
        if (h[0] != 1 || !h[5]) continue;  // PT_LOAD with memory
        uint32_t va = h[2], fsz = h[4], msz = h[5];
        uint8_t *dst = va < RAM_SIZE ? ram + va
                     : va >= ROM_BASE ? rom + (va - ROM_BASE) : drom + (va - DROM_BASE);
        memcpy(dst, b + h[1], fsz);
        memset(dst + fsz, 0, msz - fsz);
    }
    pc = entry;
    free(b);
}

static int32_t sx(uint32_t v, int bits) { return (int32_t)(v << (32 - bits)) >> (32 - bits); }

int main(int argc, char **argv) {
    long frames = -1, shot = -1;
    const char *out = NULL;
    uint32_t vbuf = 0, pal = 0;
    int quiet = 0;
    uint64_t stop = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "-frames")) frames = atol(argv[++i]);
        else if (!strcmp(argv[i], "-shot")) shot = atol(argv[++i]), out = argv[++i];
        else if (!strcmp(argv[i], "-vbuf")) vbuf = strtoul(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "-pal")) pal = strtoul(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "-q")) quiet = 1;
        else if (!strcmp(argv[i], "-stop")) stop = strtoull(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "-touch")) touch_path = argv[++i];
    }
    atexit(touch_write);
    load_elf(argv[1]);
    x[2] = RAM_SIZE;  // sp: the host sets it at the top of RAM
    long frame = 0;
    uint64_t last = 0;
    for (;;) {
        if (icount == stop && stop) {
            uint64_t h = 1469598103934665603ull;
            for (uint32_t i = 0; i < RAM_SIZE; i++) h = (h ^ ram[i]) * 1099511628211ull;
            printf("\nstop %llu pc %08x ram %016llx x", (unsigned long long)icount, pc, (unsigned long long)h);
            for (int i = 1; i < 32; i++) printf(" %x", x[i]);
            printf("\n");
            return 0;
        }
        if (pc < ROM_BASE || pc >= ROM_BASE + ROM_SIZE || pc & 3) fault("bad jump", pc);
        uint32_t w;
        memcpy(&w, rom + (pc - ROM_BASE), 4);
        icount++;
        uint32_t op = w & 0x7f, rd = w >> 7 & 31, f3 = w >> 12 & 7, rs1 = w >> 15 & 31,
                 rs2 = w >> 20 & 31, f7 = w >> 25;
        uint32_t a = x[rs1], b = x[rs2], next = pc + 4, v = 0;
        int wr = 1;
        switch (op) {
        case 0x37: v = w & 0xfffff000; break;                                   // LUI
        case 0x17: v = pc + (w & 0xfffff000); break;                            // AUIPC
        case 0x6f:                                                               // JAL
            v = next;
            next = pc + sx((w >> 31) << 20 | (w >> 12 & 0xff) << 12 | (w >> 20 & 1) << 11 | (w >> 21 & 0x3ff) << 1, 21);
            break;
        case 0x67: v = next; next = (a + sx(w >> 20, 12)) & ~1u; break;          // JALR
        case 0x63: {                                                             // branches
            int t = f3 == 0 ? a == b : f3 == 1 ? a != b : f3 == 4 ? (int32_t)a < (int32_t)b
                  : f3 == 5 ? (int32_t)a >= (int32_t)b : f3 == 6 ? a < b : f3 == 7 ? a >= b : -1;
            if (t < 0) fault("illegal", w);
            if (t) next = pc + sx((w >> 31) << 12 | (w >> 7 & 1) << 11 | (w >> 25 & 0x3f) << 5 | (w >> 8 & 0xf) << 1, 13);
            wr = 0;
            break;
        }
        case 0x03: {                                                             // loads
            uint32_t ea = a + sx(w >> 20, 12);
            switch (f3) {
            case 0: v = sx(ld(ea, 1), 8); break;
            case 1: v = sx(ld(ea, 2), 16); break;
            case 2: v = ld(ea, 4); break;
            case 4: v = ld(ea, 1); break;
            case 5: v = ld(ea, 2); break;
            default: fault("illegal", w);
            }
            break;
        }
        case 0x23: {                                                             // stores
            uint32_t ea = a + sx((w >> 25) << 5 | (w >> 7 & 31), 12);
            if (f3 > 2) fault("illegal", w);
            st(ea, b, 1 << f3);
            wr = 0;
            break;
        }
        case 0x13: {                                                             // ALU immediate
            int32_t im = sx(w >> 20, 12);
            switch (f3) {
            case 0: v = a + im; break;
            case 2: v = (int32_t)a < im; break;
            case 3: v = a < (uint32_t)im; break;
            case 4: v = a ^ im; break;
            case 6: v = a | im; break;
            case 7: v = a & im; break;
            case 1: if (f7) fault("illegal", w); v = a << rs2; break;
            case 5: v = f7 == 0x20 ? (uint32_t)((int32_t)a >> rs2) : f7 == 0 ? a >> rs2 : (fault("illegal", w), 0); break;
            }
            break;
        }
        case 0x33:                                                               // ALU, M, Zba
            if (f7 == 1) {
                int64_t sa = (int32_t)a, sb = (int32_t)b;
                switch (f3) {
                case 0: v = a * b; break;
                case 1: v = (uint64_t)(sa * sb) >> 32; break;
                case 2: v = (uint64_t)(sa * (int64_t)(uint64_t)b) >> 32; break;
                case 3: v = ((uint64_t)a * b) >> 32; break;
                case 4: v = !b ? ~0u : (a == 0x80000000u && b == ~0u) ? a : (uint32_t)((int32_t)a / (int32_t)b); break;
                case 5: v = !b ? ~0u : a / b; break;
                case 6: v = !b ? a : (a == 0x80000000u && b == ~0u) ? 0 : (uint32_t)((int32_t)a % (int32_t)b); break;
                case 7: v = !b ? a : a % b; break;
                }
            } else if (f7 == 0x10 && (f3 == 2 || f3 == 4 || f3 == 6)) {
                v = (a << (f3 >> 1)) + b;  // sh1add, sh2add, sh3add
            } else if (f7 == 0 || (f7 == 0x20 && (f3 == 0 || f3 == 5))) {
                switch (f3) {
                case 0: v = f7 ? a - b : a + b; break;
                case 1: v = a << (b & 31); break;
                case 2: v = (int32_t)a < (int32_t)b; break;
                case 3: v = a < b; break;
                case 4: v = a ^ b; break;
                case 5: v = f7 ? (uint32_t)((int32_t)a >> (b & 31)) : a >> (b & 31); break;
                case 6: v = a | b; break;
                case 7: v = a & b; break;
                }
            } else fault("illegal", w);
            break;
        case 0x0f: wr = 0; break;                                                // FENCE
        case 0x73:                                                               // ECALL
            if (w != 0x73) fault(w == 0x100073 ? "ebreak" : "illegal", w);
            wr = 0;
            if (x[17] == 93) {
                printf("\nexit %d after %llu instructions, %ld frames\n", (int)x[10],
                       (unsigned long long)icount, frame);
                return 0;
            }
            if (x[17] != 1) fault("ecall", x[17]);
            if ((x[10] & 255) != FRAME_MARK) {
                putchar(x[10] & 255);
                break;
            }
            if (!quiet) printf("[frame %ld: %llu instructions]\n", frame, (unsigned long long)(icount - last));
            last = icount;
            if (frame == shot && vbuf && pal) {
                FILE *f = fopen(out, "wb");
                uint32_t screen = ld(vbuf, 4);
                fprintf(f, "P6 320 200 255\n");
                for (int i = 0; i < 320 * 200; i++) {
                    uint8_t c = ld(screen + i, 1);
                    fputc(ld(pal + 3 * c, 1), f), fputc(ld(pal + 3 * c + 1, 1), f), fputc(ld(pal + 3 * c + 2, 1), f);
                }
                fclose(f);
            }
            if (++frame == frames) {
                printf("\nstopped after %ld frames, %llu instructions\n", frame, (unsigned long long)icount);
                return 0;
            }
            break;
        default: fault("illegal", w);
        }
        if (wr && rd) x[rd] = v;
        pc = next;
    }
}
