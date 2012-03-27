/* ftw_command_line_parser.c */

#include <ftw_std.h>
#include <ftw_rng.h>
#include <ftw_command_line_parser.h>

extern int verbose;

FILE *instream;
double box_x=7, box_y=7, box_z=7;
int mirror_depth = 1;
double sfactor = 1.0;

char *prog_name;

void parseCommandLineOptions(int argc, char *argv[])
{
  int i=0;

  prog_name = argv[0];
  if (argc < 2) printUsage();
  instream = stdin;

  while (++i<argc)
  {
    if ((*argv[i]) != '-') instream = fopen(argv[i], "r");
    else if (!strcmp(argv[i], "-usage")) printUsage();
    else if (!strcmp(argv[i], "-v")) verbose = TRUE;
    else if (!strcmp(argv[i], "-mirror_depth")) mirror_depth = getIntParameter("mirror_depth", argv[++i]);
    else if (!strcmp(argv[i], "-sfactor")) sfactor = getIntParameter("sfactor", argv[++i]);
    else if (!strcmp(argv[i], "-randomize")) randomize();
    else if (!strcmp(argv[i], "-box"))
    {
      box_x = getDoubleParameter("box_x", argv[++i]);
      box_y = getDoubleParameter("box_y", argv[++i]);
      box_z = getDoubleParameter("box_z", argv[++i]);
    }
  }
} 

void printUsage()
{
  printf("usage:  %s [-options]\n", prog_name);
  exit(0);
}
