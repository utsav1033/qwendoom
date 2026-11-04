// Doom's video layer for the Qwenputer machine, in place of doomgeneric's i_video.c.
//
// The machine has no display: Doom draws its 320 x 200 8-bit screen (I_VideoBuffer) in RAM as
// always, and at the end of each frame I_FinishUpdate prints FRAME_MARK. The host, seeing that
// byte in the trace, reads the screen and the palette (qd_palette, 768 bytes: Doom's PLAYPAL
// entry as set, gamma 0) out of the machine's memory, as a video card's scan-out would.
#include <limits.h>
#include <string.h>

#include "doomtype.h"
#include "i_video.h"
#include "rt.h"
#include "v_video.h"
#include "z_zone.h"

#define FRAME_MARK 12  // form feed: not printed by anything else

byte *I_VideoBuffer = NULL;
boolean screensaver_mode = false;
boolean screenvisible;
float mouse_acceleration = 2.0;
int mouse_threshold = 10;
int usegamma = 0;
int usemouse = 0;
int screen_width = SCREENWIDTH, screen_height = SCREENHEIGHT, screen_bpp = 8;
int fullscreen = 0, aspect_ratio_correct = 0, show_diskicon = 0, diskicon_readbytes = 0;
char *video_driver = "";

unsigned char qd_palette[768];
volatile unsigned qd_frames;  // frames finished, for the host

void I_InitGraphics(void) {
    I_VideoBuffer = (byte *)Z_Malloc(SCREENWIDTH * SCREENHEIGHT, PU_STATIC, NULL);
    screenvisible = true;
}
void I_ShutdownGraphics(void) { Z_Free(I_VideoBuffer); }
void I_StartFrame(void) {}
void I_StartTic(void) {}  // no input: a demo plays itself
void I_UpdateNoBlit(void) {}
void I_FinishUpdate(void) {
    qd_frames++;
    rv_putchar(FRAME_MARK);
}
void I_ReadScreen(byte *scr) { memcpy(scr, I_VideoBuffer, SCREENWIDTH * SCREENHEIGHT); }
void I_SetPalette(byte *palette) { memcpy(qd_palette, palette, 768); }
int I_GetPaletteIndex(int r, int g, int b) {
    int best = 0, best_diff = INT_MAX;
    for (int i = 0; i < 256; i++) {
        int dr = r - qd_palette[3 * i], dg = g - qd_palette[3 * i + 1], db = b - qd_palette[3 * i + 2];
        int d = dr * dr + dg * dg + db * db;
        if (d < best_diff) best = i, best_diff = d;
    }
    return best;
}
void I_BeginRead(void) {}
void I_EndRead(void) {}
void I_SetWindowTitle(char *title) { (void)title; }
void I_GraphicsCheckCommandLine(void) {}
void I_SetGrabMouseCallback(grabmouse_callback_t func) { (void)func; }
void I_EnableLoadingDisk(void) {}
void I_BindVideoVariables(void) {}
void I_DisplayFPSDots(boolean dots_on) { (void)dots_on; }
void I_CheckIsScreensaver(void) {}
void I_InitWindowTitle(void) {}
void I_InitWindowIcon(void) {}
