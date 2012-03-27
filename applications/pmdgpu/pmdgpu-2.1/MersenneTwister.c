/* MersenneTwister.c */

/***********************************************************************/
/*                                                                     */
/* Standard random number generator, initialized with a separate seed  */
/* value, creates a reproducible sequence of pseudo-random numbers     */
/* which is unique to each thread.                                     */
/*                                                                     */
/***********************************************************************/

#include <pmdgpu.h>

void MersenneInitialize(struct MersenneTwister *p_t, int seed)
{
  int i;
  p_t->mt[0] = seed;
  for (p_t->mti=1; p_t->mti<MERSENNE_NN; p_t->mti++) p_t->mt[p_t->mti] =  (6364136223846793005ULL * (p_t->mt[p_t->mti-1] ^ (p_t->mt[p_t->mti-1] >> 62)) + p_t->mti);

  // run it a few times to randomize it
  for (i=0; i<100000; i++) rndm(p_t);
}

float rndm(struct MersenneTwister* MT)
{
    int i;
    unsigned long long x;
    static unsigned long long mag01[2]={0ULL, MERSENNE_MATRIX_A};

    if (MT->mti >= MERSENNE_NN) { /* generate NN words at one time */

      /* if init_genrand64() has not been called, */
      /* a default initial seed is used     */
      //if (MT->mti == MERSENNE_NN+1) 
      //    init_genrand64(5489ULL); 

      for (i=0;i<MERSENNE_NN-MERSENNE_MM;i++) {
          x = (MT->mt[i]&MERSENNE_UM)|(MT->mt[i+1]&MERSENNE_LM);
          MT->mt[i] = MT->mt[i+MERSENNE_MM] ^ (x>>1) ^ mag01[(int)(x&1ULL)];
      }
      for (;i<MERSENNE_NN-1;i++) {
          x = (MT->mt[i]&MERSENNE_UM)|(MT->mt[i+1]&MERSENNE_LM);
          MT->mt[i] = MT->mt[i+(MERSENNE_MM-MERSENNE_NN)] ^ (x>>1) ^ mag01[(int)(x&1ULL)];
      }
      x = (MT->mt[MERSENNE_NN-1]&MERSENNE_UM)|(MT->mt[0]&MERSENNE_LM);
      MT->mt[MERSENNE_NN-1] = MT->mt[MERSENNE_MM-1] ^ (x>>1) ^ mag01[(int)(x&1ULL)];

      MT->mti = 0;
  }
  
  x = MT->mt[MT->mti++];

  x ^= (x >> 29) & 0x5555555555555555ULL;
  x ^= (x << 17) & 0x71D67FFFEDA60000ULL;
  x ^= (x << 37) & 0xFFF7EEE000000000ULL;
  x ^= (x >> 43);

  return (x >> 11) * (1.0/9007199254740992.0);
}
