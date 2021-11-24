/* ftw_std.h */

#ifndef FTW_STD_INCLUDE
#define FTW_STD_INCLUDE

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

int verbose;

#ifndef V
#define V if (verbose)
#endif

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

double getDoubleParameter(char *param, char *value);
int getIntParameter(char *param, char *value);

#endif
