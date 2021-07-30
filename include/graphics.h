/* graphics.h */

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xos.h>
#include <X11/keysym.h>
#include <X11/Xresource.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void checkForWindowEvent();
void initializeDisplay();
void setChangeFlag();
void resetChangeFlag();
int changeFlagIsSet();
int graphicsModeEnabled();
void drawGraphicalRepresentation();
