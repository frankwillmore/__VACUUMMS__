/* ftw_gfg2fvi.h */

#include <ftw_types.h>

#ifndef FTW_GFG2FVI_H
#define FTW_GFG2FVI_H

//  These are the routines to call from outside the library
//  They are conditionally declared extern because this header is included by the .cu file 
//  which is compiled by nvcc and which mangles linkage by default.  
//  The libraries thus generated will be linkable from C as well as C++

#ifdef __cplusplus
extern "C" 
#endif
ftw_FVI256 *gfgToFVI256(ftw_GFG65536 *gfg, float sigma, float epsilon);

#ifdef __cplusplus
extern "C" 
#endif
ftw_EnergyArray256 *GFGToEnergyArray256(ftw_GFG65536 *gfg, float sigma, float epsilon);




#ifdef __cplusplus
extern "C" 
#endif
ftw_EnergyArray256 *GFGToRepulsion256(ftw_GFG65536 *gfg, float sigma, float epsilon);

#ifdef __cplusplus
extern "C" 
#endif
ftw_EnergyArray256 *GFGToRepulsion256_612(ftw_GFG65536 *gfg, float sigma, float epsilon);



#ifdef __cplusplus
extern "C" 
#endif
ftw_EnergyArray512 *GFGToRepulsion512(ftw_GFG65536 *gfg, float sigma, float epsilon);

#ifdef __cplusplus
extern "C" 
#endif
ftw_EnergyArray512 *GFGToRepulsion512_612(ftw_GFG65536 *gfg, float sigma, float epsilon);



/*
#ifdef __cplusplus
extern "C" 
#endif
ftw_EnergyArray1024 *GFGToEnergyArray1024_69(ftw_GFG65536 *gfg, float sigma, float epsilon);

#ifdef __cplusplus
extern "C" 
#endif
ftw_EnergyArray1024 *GFGToRepulsion1024_69(ftw_GFG65536 *gfg, float sigma, float epsilon);
*/

#ifdef __cplusplus
extern "C" 
#endif
ftw_EnergyArray1024 *GFGToEnergyArray1024_612(ftw_GFG65536 *gfg, float sigma, float epsilon);

#ifdef __cplusplus
extern "C" 
#endif
ftw_EnergyArray1024 *GFGToRepulsion1024_612(ftw_GFG65536 *gfg, float sigma, float epsilon);

struct Chunk
{
  float energy[256][1024][1024];
};

typedef struct Chunk ftw_Chunk;

#endif // FTW_GFG2FVI_H

