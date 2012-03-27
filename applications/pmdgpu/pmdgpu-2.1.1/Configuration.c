/* Configuration.cu */

/*************************************************************/
/*                                                           */
/* Reads a file with five fields of tab separated values.    */
/* (x, y, z, sigma, epsilon) for Lennard-Jones 6-9           */
/* non-bond interaction (COMPASS Forcefield)                 */
/*                                                           */
/* stores the parameters as cross-interaction with the       */
/* test particle via the interaction specified in:           */
/*   "COMPASS Force Field for 14 Inorganic Molecules         */
/*    Jie Yang, Yi Ren, and An-min Tian                      */
/*    J. Phys. Chem. B 2000, 104, 4951-4957"                 */
/*                                                           */
/*************************************************************/

#include <pmdgpu.h>

extern struct Atom configuration[];
extern long number_of_atoms;
extern struct CommandLineOptions clo;

extern int mpi_rank;

void readConfiguration()
{
  FILE *in=stdin;
  char line[80];
  char *xs, *ys, *zs;
  char *sigmas, *epsilons;
  char filename[256];

  float r_j; // i is test molecule, j is configuration molecule
  float epsilon_j;

  if (!clo.use_stdin)
  {
    sprintf(filename, "%s/%s%d%s", clo.config_directory, clo.config_prefix, clo.config_start + mpi_rank, clo.config_suffix);
    printf("opening %s for configuration input data...\n", filename);
    in = fopen(filename, "r");
    if (in == NULL) 
    {
      printf("couldn't open %s.  exiting...\n"); 
      exit(1);
    }
  }

  while (1)
  {
    fgets(line, 80, in);
    if (feof(in)) break;

    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");
    sigmas = strtok(NULL, "\t");
    epsilons = strtok(NULL, "\n");

    configuration[number_of_atoms].x = strtod(xs, NULL);
    configuration[number_of_atoms].y = strtod(ys, NULL);
    configuration[number_of_atoms].z = strtod(zs, NULL);

    r_j = strtod(sigmas, NULL);
    epsilon_j = 503.25f * strtod(epsilons, NULL); // convert from kcal/mol to K

    configuration[number_of_atoms].r_ij = pow(0.5f * (pow(clo.r_i, 6) + pow(r_j, 6)), 0.1666666f);
    configuration[number_of_atoms].r_ij1 = pow(0.5f * (pow(clo.r_i1, 6) + pow(r_j, 6)), 0.1666666f);
    configuration[number_of_atoms].r_ij2 = pow(0.5f * (pow(clo.r_i2, 6) + pow(r_j, 6)), 0.1666666f);
    configuration[number_of_atoms].epsilon_ij = 2.0 * sqrt(clo.epsilon_i * epsilon_j) * (pow(clo.r_i, 3) * pow(r_j, 3)) / (pow(clo.r_i, 6) + pow(r_j, 6));
    configuration[number_of_atoms].epsilon_ij1 = 2.0 * sqrt(clo.epsilon_i1 * epsilon_j) * (pow(clo.r_i1, 3) * pow(r_j, 3)) / (pow(clo.r_i1, 6) + pow(r_j, 6));
    configuration[number_of_atoms].epsilon_ij2 = 2.0 * sqrt(clo.epsilon_i2 * epsilon_j) * (pow(clo.r_i2, 3) * pow(r_j, 3)) / (pow(clo.r_i2, 6) + pow(r_j, 6));
//printf("%f\t", configuration[number_of_atoms].epsilon_ij);
//printf("%f\t", configuration[number_of_atoms].epsilon_ij1);
//printf("%f\n", configuration[number_of_atoms].epsilon_ij2);

    number_of_atoms++;
  }
 
  fclose(in);
}
