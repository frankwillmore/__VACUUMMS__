/* ftw_rng2.h */

/* random number generator will return 0 <= rnd() < 1 or 0 <= rnd_int() < 2^32 */

#ifndef FTW_RNG2_INCLUDE
#define FTW_RNG2_INCLUDE

/* How this works:  if seed = -1, generator is seeded from system clock */
/*                  if seed = 0, generator is seeded with default value */
/*                  if seed > 0, generator is seeded with value provided */

void initializeRandomNumberGenerator2(int seed);
double rnd2();
int rnd2Int();

#endif

