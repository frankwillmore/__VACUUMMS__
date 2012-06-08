// test.c

#include <ftw_types.h>
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <assert.h>

#include "ftw_prng.h"

struct MersenneTwister *rng;

void *ThreadMain(void *arg)
{ 
  int j;
  int tid=*((int*)arg);

  MersenneInitialize(&rng[tid], tid);
  printf("thread #%d\t", tid);
  for (j=0; j<10; j++) printf("%lf\t", prnd(&rng[tid]));
  printf("\n");
  return NULL;
}

int main()
{
  int num_threads;
  scanf ("%d", &num_threads);

  pthread_t threads[num_threads];
  int thread_args[num_threads];
  int rc, i;
 
  rng = (struct MersenneTwister *)malloc(sizeof(struct MersenneTwister) * num_threads);

  /* create all threads */
  for (i=0; i<num_threads; ++i) {
    thread_args[i] = i;
    printf("In main: creating thread %d\n", i);
    rc = pthread_create(&threads[i], NULL, ThreadMain, (void *) &thread_args[i]);
    assert(0 == rc);
  }
 
  /* wait for all threads to complete */
  for (i=0; i<num_threads; ++i) {
    rc = pthread_join(threads[i], NULL);
    assert(0 == rc);
  }

  free(rng);
  return 0;
}

