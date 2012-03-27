/*********************** ftw_rng.c *********************************/
#include "ftw_rng.h"
#include <math.h>
#include <time.h>
#include <stdio.h>

#define thirtyone 2147483648.0
#ifndef DEFAULT_SEED
#define DEFAULT_SEED 12345
#endif

int ra[256], nd;

void initializeRandomNumberGeneratorTo(int seed)
{
  randinit(seed);
}

void initializeRandomNumberGenerator()
{
  randinit(DEFAULT_SEED);
}

void randinit(int seed)
{
  double dubee, e;
  int trune;
  long int i;
  dubee=-1.0+1.0/thirtyone;
  e = seed/thirtyone;
  for (i=1; i<10000; i++)
  {
    e*=16807;
    trune=e;
    e+=dubee*trune;
    if (e>=1) e+= dubee;
  }
  for (nd=0;nd<=255; nd++)
  {
    e=16807*e;
    trune=e;
    e+=dubee*trune;
    if (e>=1) e+=dubee;
    ra[nd]=(thirtyone)*e;
  }
  for (i=0; i<=10000; i++)
  {
    nd=(nd+1)&255;
    ra[nd]=(ra[(nd-103)&255])^(ra[(nd-250)&255]);
  }
} /*end randinit*/

double RND()
{
  return rnd();
} /* end RND */

double rnd()
{
  nd=(nd+1)&255;
  ra[nd]=(ra[(nd-103)&255]^ra[(nd-250)&255]);
  return ra[nd]/thirtyone;
} 

int random_int(int max)
{
  return floor(rnd() * max);
}

double random_double(double max)
{
  return (rnd() * max);
}

/* randomly initializes rng */
int randomize()
{
  time_t now;
  struct tm *p_tyme;
  struct tm tyme;
  int rng_seed;

  now = time(NULL);
  p_tyme = gmtime(&now);
  tyme = *p_tyme;
  rng_seed = tyme.tm_sec * tyme.tm_min * tyme.tm_hour;
  initializeRandomNumberGeneratorTo(rng_seed);
  return rng_seed;
}

/* returns random seed value */
int getRandomSeed()
{
  time_t now;
  struct tm *p_tyme;
  struct tm tyme;
  int rng_seed;

  now = time(NULL);
  p_tyme = gmtime(&now);
  tyme = *p_tyme;
  rng_seed = tyme.tm_sec * tyme.tm_min * tyme.tm_hour;
//printf("rng_seed = %d", rng_seed);
  return rng_seed;
}
