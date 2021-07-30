/* prng.h */

/****************************************************************************/
/*                                                                          */
/*                          Parallel RNG		                    */
/*                                                                          */
/*                        (C) 2012 Frank T Willmore                         */
/*                                                                          */
/*    Contains parallel implementation of Mersenne Twister                  */
/*                                                                          */
/*                                                                          */
/*    The Texas Advanced Computing Center                                   */
/*    The National Science Foundation/NSF-Teragrid                          */
/*                                                                          */
/*    correspondence to:  frankwillmore@gmail.com                           */
/*                                                                          */
/****************************************************************************/

#define MERSENNE_NN 312
#define MERSENNE_MM 156
#define MERSENNE_MATRIX_A 0xB5026F5AA96619E9ULL
#define MERSENNE_UM 0xFFFFFFFF80000000ULL /* Most significant 33 bits */
#define MERSENNE_LM 0x7FFFFFFFULL /* Least significant 31 bits */

struct MersenneTwister
{
  unsigned long long mt[MERSENNE_NN];
  int mti;
};

/* prototypes */

void MersenneInitialize(struct MersenneTwister* MT, int seed);
double prnd(struct MersenneTwister* MT);
