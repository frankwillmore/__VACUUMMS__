/* ftw_gfgEnergy.h */

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>

#include <ftw_types.h>

#ifndef FTW_GFGENERGY_H
#define FTW_GFGENERGY_H

//  These are the routines to call from outside the library
//  which is compiled by nvcc and which mangles linkage by default.  
//  The libraries thus generated will be linkable from C as well as C++

// routine to return energy at a single point

#ifdef __cplusplus
extern "C" 
#endif
float ftw_GFG65536Energy_69(ftw_GFG65536 *gfg, float x, float y, float z);

#ifdef __cplusplus
extern "C" 
#endif
float ftw_GFG65536Energy_612(ftw_GFG65536 *gfg, float x, float y, float z, float diameter);


#ifdef __cplusplus
extern "C" 
#endif
int ftw_GFG65536HS_Overlap(ftw_GFG65536 *gfg, float x_j, float y_j, float z_j, float diameter);



#endif

