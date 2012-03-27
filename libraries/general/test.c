// test.c

#include <ftw_types.h>
#include <ftw_config_parser.h>
#include <stdio.h>


main()
{
  int i;

  ftw_Configuration *cfg = readGFG(stdin);
  ftw_Configuration *rfg = replicateGFG(cfg, 10, 10, 10);

  for (i=0; i<rfg->n_atoms; i++) printf("%lf\t%f\n", rfg->atom[i].x, rfg->atom[i].y); 
}
