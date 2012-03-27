#include <genels.h>

extern char els_directory[];

void generateOutput(struct ZThread* thread)
{
  FILE *fptr;
  char filename[80];

  sprintf(filename, "%s/%0X%0X%0X.els", els_directory, thread->i, thread->j, thread->k);

  printf("opening %s for output.\n\n", filename);

  fptr = fopen(filename, "w");
  if(fptr == NULL) 
  {
    perror("could not open output file");
    return;
  }

  fwrite((void *)(thread->energy_array), sizeof(struct EnergyArray), 1, fptr);

  fclose(fptr);
}
