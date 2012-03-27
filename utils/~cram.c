/* cram.c */

#define MAX_CAVITIES 16384
#define N_POINTS 10000

#include <ftw_std.h>
#include <ftw_rng2.h>
#include <ftw_param.h>

// Reads a centered cluster and determines the radius of gyration
// Will deliver an erroneous result if cluster is not centered or percolates.

// Mechanism:  Samples points in space to see if they are part of cluster
//             If they are, they are added to a set of points used in
//             determining the radius of gyration.

// In:  .cav 
// Out: .dst (reports one value)

FILE *instream;

int main(int argc, char* argv[])
{
  int i,j;
  double box_x, box_y, box_z;
  double distance;
  double max_distance = 0;
  double x, y, z, sigma, epsilon;
  char line[80];
  char *xs, *ys, *zs, *sigmas, *epsilons;

  instream=stdin;

  setCommandLineParameters(argc, argv);
  getVectorParam("-box", &box_x, &box_y, &box_z);

  if (getFlagParam("-usage"))
  {
    printf("\n");
    printf("usage:\t-box [ nn.nnn nn.nnn nn.nnn ]\n");
    printf("      \tgfg in, gfg out - crammed into the box by PBC\n\n");
    exit(0);
  }

  // This doesn't work...
  if (box_x * box_y * box_z == 0) 
  {
    printf("Did you specify box size?\n");
    exit(1);
  }

  while (TRUE)
  {
    fgets(line, 80, instream);
    if (feof(instream)) break;

    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");
    sigmas = strtok(NULL, "\t");
    epsilons = strtok(NULL, "\n");

    x = strtod(xs, NULL);
    y = strtod(ys, NULL);
    z = strtod(zs, NULL);
    sigma = strtod(sigmas, NULL);
    epsilon = strtod(epsilons, NULL);

    while (x < 0) x+= box_x;
    while (x > box_x) x-= box_x;

    while (y < 0) y+= box_y;
    while (y > box_y) y-= box_y;

    while (z < 0) z+= box_z;
    while (z > box_z) z-= box_z;

    printf("%lf\t%lf\t%lf\t%lf\t%lf\n", x, y, z, sigma, epsilon);
  }
}

