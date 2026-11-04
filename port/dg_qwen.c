// doomgeneric's platform functions for the Qwenputer machine, and main.
//
// The clock is a counter, not a timer: one game tic (1000 / 35 ms) per finished frame, plus any
// sleep Doom asks for, so a run is the same instruction stream every time. With -timedemo Doom
// runs one tic per frame regardless (singletics). There is no keyboard: the demo plays itself.
#include <stdint.h>

#include "doomgeneric.h"
#include "m_argv.h"

extern volatile unsigned qd_frames;
static uint32_t slept;

pixel_t *DG_ScreenBuffer = NULL;  // unused: the host reads Doom's own 8-bit screen
void DG_Init(void) {}
void DG_DrawFrame(void) {}
void DG_SleepMs(uint32_t ms) { slept += ms; }
uint32_t DG_GetTicksMs(void) { return qd_frames * 29 + slept; }
int DG_GetKey(int *pressed, unsigned char *key) {
    (void)pressed, (void)key;
    return 0;
}
void DG_SetWindowTitle(const char *title) { (void)title; }

void D_DoomMain(void);
void doomgeneric_Tick(void);

static char *args[] = {"doom", "-iwad", "doom.wad", "-mb", "3", "-timedemo", "demo1", NULL};

int main(void) {
    myargc = sizeof args / sizeof *args - 1;
    myargv = args;
    D_DoomMain();
    for (;;) doomgeneric_Tick();
}
