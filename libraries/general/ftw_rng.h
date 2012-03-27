/* ftw_rng.h */

#ifndef FTW_RNG_INCLUDE
#define FTW_RNG_INCLUDE

void initializeRandomNumberGeneratorTo(int seed);
void initializeRandomNumberGenerator();
void randinit(int seed);
double RND();
double rnd();
int randomize();
double rnd_double();
int rnd_int();
int getRandomSeed();

#endif

