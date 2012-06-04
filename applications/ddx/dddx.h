/* dddx.h */

#include <ftw_types.h>
#include <pthread.h>

//double calculateRepulsion();

typedef struct dddCavity
{
  int i;
  int j;
  int k;
  float x;
  float y;
  float z;
  float diameter;
  struct Cavity *next;
  ftw_GFG65536 *gfg;
  pthread_t *thread;
} dddCavity;

void *expandTestParticle(void *passval);
float calculateEnergy(dddCavity *cavity, float diameter);
