/* ftw_config_parser.c */

//  A set of library routines to parse (GFG) input file

#include <stdlib.h>
#include <stdio.h>
#include <ftw_param.h>
#include <string.h>
#include <ftw_types.h>

#include <ftw_config_parser.h>

// IN:	input stream (or file pointer)
// OUT: configuration data stored as *ftw_GFG65536 
ftw_GFG65536 *readGFG65536(FILE *instream)
{
  int lines_read=0;
  char line[80];
  char *xs, *ys, *zs, *ds, *es;

  // allocate the return structure... 
  ftw_GFG65536 *configuration = (ftw_GFG65536*)malloc(sizeof(ftw_GFG65536));

  for (lines_read=0; ; lines_read++)
  {
    fgets(line, 80, instream);

    if (feof(instream)) break;
    
    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");
    ds = strtok(NULL, "\t");
    es = strtok(NULL, "\n");

    configuration->atom[lines_read].x = strtod(xs, NULL);
    configuration->atom[lines_read].y = strtod(ys, NULL);
    configuration->atom[lines_read].z = strtod(zs, NULL);
    configuration->atom[lines_read].sigma = strtod(ds, NULL);
    configuration->atom[lines_read].epsilon = strtod(es, NULL);
  }
 
  fclose(instream);
  configuration->n_atoms = lines_read; 
  return configuration;
}

// IN:	set of Atoms, and a pointer to hold the replicated set of Atoms.
// OUT:	replicated set of 27x atoms.  Uses the dimensions of the original box.
ftw_GFG65536 *replicateGFG65536(ftw_GFG65536 *in)
{
  int i, j, k;
  int n;
  long n_out=0;

  ftw_GFG65536 *configuration = (ftw_GFG65536*)malloc(sizeof(ftw_GFG65536));
  
  for (i=-1; i<=1; i++)
  for (j=-1; j<=1; j++)
  for (k=-1; k<=1; k++)
  for (n=0; n<in->n_atoms; n++)
  {
    configuration->atom[n_out].x = in->atom[n].x + i * in->box_x;
    configuration->atom[n_out].y = in->atom[n].y + j * in->box_y;
    configuration->atom[n_out].z = in->atom[n].z + k * in->box_z;
    configuration->atom[n_out].sigma   = in->atom[n].sigma;
    configuration->atom[n_out].epsilon = in->atom[n].epsilon;
    n_out++; // total # of atoms in replica
  }

  configuration->n_atoms = n_out;

  // Use the input box dimensions
  configuration->box_x = in->box_x;
  configuration->box_y = in->box_y;
  configuration->box_z = in->box_z;
  return configuration;
}

