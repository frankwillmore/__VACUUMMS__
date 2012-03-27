#include "pmdgpu.h"

main()
{
  struct MersenneTwister mt;
  int i;

  MersenneInitialize(&mt, 5);
  for (i=0; i<1000; i++) 
  {
    //printf("%f\n", rndm(&mt));
    //printf("%f\n", -log(rndm(&mt)));
    printf("%f\n", acos(1.0f-2.0f*rndm(&mt)));
  }
}

