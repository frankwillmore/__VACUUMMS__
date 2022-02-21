/* CommandLineOptions.c */

#include <genels.h>

extern float r_i;
extern float epsilon_i;
extern float verlet_cutoff_sq;
extern float box_dimension;
extern int concurrency;
extern char els_directory[];

void readCommandLineOptions(int argc, char* argv[])
{
  int i=0;

  while (++i<argc)
  {
    if (!strcmp(argv[i], "-usage")) 
    {
      printf("\nusage:  .gfg configuration in, directory of els blocks out\n\n");
      printf("\n");
      exit(0);
    }
    else if (!strcmp(argv[i], "-verlet_cutoff_sq")) verlet_cutoff_sq = (strtod(argv[++i], NULL));
    else if (!strcmp(argv[i], "-r_i")) r_i = (strtod(argv[++i], NULL));
    else if (!strcmp(argv[i], "-epsilon_i")) epsilon_i = (strtod(argv[++i], NULL));
    else if (!strcmp(argv[i], "-concurrency")) concurrency = (strtol(argv[++i], NULL, 10));
    else if (!strcmp(argv[i], "-box_dimension")) box_dimension = (strtod(argv[++i], NULL));
    else if (!strcmp(argv[i], "-els_directory"))
    {
      strcpy(els_directory, argv[++i]);
      if(mkdir(argv[i], "777"))
      {
        if (errno != EEXIST)
        {
          printf("Could not create directory %s.\nExiting...\n\n", argv[i]);
          exit(1);
        }
      }
      printf("Created directory %s.\n", argv[i]);
    }
  }

  printf("\n\nProcessed all command line options.\n");
  printf("Using:  r_i =\t%f\n", r_i);
  printf("        epsilon_i =\t%f\n", epsilon_i);
  printf("        verlet_cutoff_sq =\t%f\n", verlet_cutoff_sq);
  printf("        box_dimension =\t%f\n", box_dimension);
  printf("        concurrency =\t%d\n", concurrency);
  printf("        els_directory =\t%s\n", els_directory);
  printf("\n\n");

  fflush(stdout);

} /* end CommandLineOptions */

