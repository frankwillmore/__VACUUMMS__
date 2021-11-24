/* ftw_cav_parser.c */

//  A set of library routines to parse (CAV) input file

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <ftw_param.h>
#include <ftw_types.h>

#include <ftw_cav_parser.h>

// IN:	input stream (or file pointer)
// OUT: cavities data stored as *ftw_CAV65536 
ftw_CAV65536 *readCAV65536(FILE *instream)
{
  int lines_read=0;
  char line[80];
  char *xs, *ys, *zs, *ds;

  // allocate the return structure... 
  ftw_CAV65536 *cavities = (ftw_CAV65536*)malloc(sizeof(ftw_CAV65536));

  for (lines_read=0; ; lines_read++)
  {
    fgets(line, 80, instream);

    if (feof(instream)) break;
    
    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");
    ds = strtok(NULL, "\n");

    cavities->cavity[lines_read].x = strtod(xs, NULL);
    cavities->cavity[lines_read].y = strtod(ys, NULL);
    cavities->cavity[lines_read].z = strtod(zs, NULL);
    cavities->cavity[lines_read].diameter = strtod(ds, NULL);
  }
 
  fclose(instream);
  cavities->n_cavities = lines_read; 
  return cavities;
}

// IN:	set of Atoms, and a pointer to hold the replicated set of Atoms.
// OUT:	replicated set of 27x cavity.  Uses the dimensions of the original box.
ftw_CAV65536 *replicateCAV65536(ftw_CAV65536 *in)
{
  int i, j, k;
  int n;
  long n_out=0;

  ftw_CAV65536 *cavities = (ftw_CAV65536*)malloc(sizeof(ftw_CAV65536));
  
  for (i=-1; i<=1; i++)
  for (j=-1; j<=1; j++)
  for (k=-1; k<=1; k++)
  for (n=0; n<in->n_cavities; n++)
  {
    cavities->cavity[n_out].x = in->cavity[n].x + i * in->box_x;
    cavities->cavity[n_out].y = in->cavity[n].y + j * in->box_y;
    cavities->cavity[n_out].z = in->cavity[n].z + k * in->box_z;
    cavities->cavity[n_out].diameter   = in->cavity[n].diameter;
    n_out++; // total # of cavity in replica
  }

  cavities->n_cavities = n_out;

  // Use the input box dimensions
  cavities->box_x = in->box_x;
  cavities->box_y = in->box_y;
  cavities->box_z = in->box_z;
  return cavities;
}

