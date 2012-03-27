#include <genels.h>

main(int argc, char* argv[])
{
  int i;
  FILE *fptr;
  char filename[80];
  struct EnergyArray nrgra;

  sprintf(filename, argv[1]);
printf("oopening %s...\n", filename);
  
  fptr = fopen(filename, "r");
  if(fptr == NULL) {
        perror("failed to open sample.txt");
        return;
    }

  fread((void*)&nrgra, sizeof(struct EnergyArray), 1, fptr);
  fclose(fptr);

  for (i=0; i<FINE_GRID_RESOLUTION; i++) printf("%f\n", nrgra.energy[i][i][i]);
}
