#include "graphics.c"
#include "rng.c"

int main(int argc, char *argv[])
{
  startgraphics();
  
  while (1)
  {
    gcvalues.foreground = BG_COLOR;
    XChangeGC(dpy, context, GCForeground, &gcvalues);
    XFillRectangle(dpy, window, context, 0, 0, WSIZE*2, WSIZE*2);
  
    gcvalues.foreground = FG_COLOR;
    XChangeGC(dpy, context, GCForeground, &gcvalues);
    XFillRectangle(dpy, window, context, 10, 10, 20, 20);

    check4event();
  
    XFillArc(dpy, window, context, 25, 50, 30, 15, 0, 360*64);
  }
}

