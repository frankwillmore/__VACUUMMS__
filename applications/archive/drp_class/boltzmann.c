/* boltzmann.c*/

#include "rng.c"

int main()
{
  int i=0, j=0;
  int N=100,W=30;
  int trials=1000;

  int slots[N];
  int omega[W];

  for (i=0; i<N; i++) slots[i]=0;
  for (j=0; j<W; j++) omega[j]=0;

  randinit();

  for (i=0; i<trials; i++)
  {
    int n = RND()*N;
    slots[n]++;
    //printf("picking %d\n", n);
  }

  for (i=0; i<100; i++)
  {
    omega[slots[i]]++;
    //printf ("slot %d has %d\n", i, slots[i]);
  }

  for (j=0; j<W; j++)
  {
    printf("%d\t%d\n", j, omega[j]);
  }    
}
