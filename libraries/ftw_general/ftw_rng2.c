/*********************** ftw_rng2.c *********************************/
#include <ftw_rng2.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define thirtyone 2147483648.0
#ifndef DEFAULT_SEED
#define DEFAULT_SEED 12345
#endif

int ra2[256], nd2;

int RNG_IS_INITIALIZED=0;

void initializeRandomNumberGenerator2(int seed)
{
  double dubee, e;
  int trune;
  long int i;
  
  time_t now;
  struct tm *p_tyme;
  struct tm tyme;

  RNG_IS_INITIALIZED=1;

  if (seed==0) seed=DEFAULT_SEED;
  if (seed==-1) 
  {
    now = time(NULL);
    p_tyme = gmtime(&now);
    tyme = *p_tyme;
    seed = tyme.tm_hour * 3600 + tyme.tm_min * 60 + tyme.tm_sec + 123;
  }

  dubee=-1.0+1.0/thirtyone;
  e = seed/thirtyone;

  for (i=1; i<10000; i++)
  {
    e*=16807;
    trune=e;
    e+=dubee*trune;
    if (e>=1) e+= dubee;
  }
  for (nd2=0;nd2<=255; nd2++)
  {
    e=16807*e;
    trune=e;
    e+=dubee*trune;
    if (e>=1) e+=dubee;
    ra2[nd2]=(thirtyone)*e;
  }
  for (i=0; i<=10000; i++)
  {
    nd2=(nd2+1)&255;
    ra2[nd2]=(ra2[(nd2-103)&255])^(ra2[(nd2-250)&255]);
  }
} /*end randinit*/

double rnd2()
{
  if (!RNG_IS_INITIALIZED)
  {
    printf("call to rnd() without initializing generator\n");
    exit(1);
  }
  nd2=(nd2+1)&255;
  ra2[nd2]=(ra2[(nd2-103)&255]^ra2[(nd2-250)&255]);
  return ra2[nd2]/thirtyone;
} 

int rnd2Int()
{
  if (!RNG_IS_INITIALIZED)
  {
    printf("call to rnd() without initializing generator\n");
    exit(1);
  }
  nd2=(nd2+1)&255;
  nd2=(nd2+1)&255;
  ra2[nd2]=(ra2[(nd2-103)&255]^ra2[(nd2-250)&255]);
  return ra2[nd2];
}

