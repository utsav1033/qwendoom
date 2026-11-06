#include <stdio.h>
#include <stdlib.h>

// The Mandelbrot set in text, in c4's C: no floats, so numbers are fixed point (4096 = 1.0).

int W, H, ITER;

int escape(int cr, int ci)
{
  int zr, zi, zr2, zi2, i;
  zr = 0; zi = 0; zr2 = 0; zi2 = 0; i = 0;
  while (i < ITER && zr2 + zi2 < 4 * 4096) {
    zi = 2 * zr * zi / 4096 + ci;
    zr = zr2 - zi2 + cr;
    zr2 = zr * zr / 4096;
    zi2 = zi * zi / 4096;
    i++;
  }
  return i;
}

int main()
{
  int x, y, n;
  char *ramp, *line;
  W = 40; H = 16; ITER = 16;
  ramp = " .:-=+*#%@";
  line = malloc(W + 1);
  y = 0;
  while (y < H) {
    x = 0;
    while (x < W) {
      n = escape(x * 3 * 4096 / W - 2 * 4096 - 2048, y * 2 * 4096 / H - 4096 + 4096 / H);
      line[x] = n == ITER ? '@' : ramp[n * 9 / ITER];
      x++;
    }
    line[W] = 0;
    printf("%s\n", line);
    y++;
  }
  return 0;
}
