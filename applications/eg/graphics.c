#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xos.h>
#include <X11/keysym.h>
#include <X11/Xresource.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "graphics.h"

Display *dpy;
Window window;
Window window2;
GC context;
GC context2;
XEvent event;
int rstat;
XGCValues gcvalues;
char *displayname="simulation";
char *display=NULL;
extern int side_view;
extern int wsize_x, wsize_y, wsize_z;
char *display_name_1 = "X-Y projection (front)";
char *display_name_2 = "Z-Y projection (right side)";

void check4event()
{
  rstat=XCheckMaskEvent(dpy, ExposureMask | ButtonPressMask, &event);
  if (rstat&&(event.type == ButtonPress))
  {
    XFreeGC(dpy, context);
    XCloseDisplay(dpy);
    exit(0);
  }
}

void startGraphicsXYZ()
{
  dpy = XOpenDisplay(display);
  if (dpy == NULL) printf("Can't open the display.\n");
  //assert(dpy);
 
  window = XCreateSimpleWindow(dpy, DefaultRootWindow(dpy), 0, 0, wsize_x, wsize_y, 0,0,0);
  XSelectInput(dpy, window, (StructureNotifyMask|ExposureMask|ButtonPressMask|ButtonReleaseMask));
  XMapWindow(dpy, window);
  context = XCreateGC(dpy, window, GCForeground | GCBackground, &gcvalues);
  XStoreName(dpy, window, display_name_1);
  XSetIconName(dpy, window, display_name_2);

  if (side_view)
  {
    window2 = XCreateSimpleWindow(dpy, DefaultRootWindow(dpy), wsize_x, 0, wsize_z, wsize_y, 0,0,0);
    XSelectInput(dpy, window2, (StructureNotifyMask|ExposureMask|ButtonPressMask|ButtonReleaseMask));
    XMapWindow(dpy, window2);
    context2 = XCreateGC(dpy, window2, GCForeground | GCBackground, &gcvalues);
    XStoreName(dpy, window2, display_name_2);
    XSetIconName(dpy, window2, display_name_2);
  }

  check4event();
  gcvalues.background = 0;
}

void startgraphics(int wsize)
{
  dpy = XOpenDisplay(display);
  if (dpy == NULL) printf("Can't open the display.\n");
  //assert(dpy);
  
  window = XCreateSimpleWindow(dpy, DefaultRootWindow(dpy), 10, 10, wsize, wsize, 0,0,0);
  XSelectInput(dpy, window, (StructureNotifyMask|ExposureMask|ButtonPressMask|ButtonReleaseMask));
  XMapWindow(dpy, window);
  context = XCreateGC(dpy, window, GCForeground | GCBackground, &gcvalues);
  XStoreName(dpy, window, displayname);
  XSetIconName(dpy, window, displayname);
  check4event();
  gcvalues.background = 0;
}

