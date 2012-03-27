/* readPolymerConfiguration.c */

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

#include <genels.h>

extern struct Atom configuration[];
extern int number_of_atoms;
extern float box_dimension;
extern float r_i;
extern float epsilon_i;

void readPolymerConfiguration()
{
  char line[80];
  char *xs, *ys, *zs;
  char *sigmas, *epsilons;
  int replica_x, replica_y, replica_z;

  float r_j; // i is test molecule, j is configuration molecule
  float epsilon_j;

  while (1)
  {
    fgets(line, 80, stdin);
    if (feof(stdin)) break;

    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");
    sigmas = strtok(NULL, "\t");
    epsilons = strtok(NULL, "\n");

    r_j = strtod(sigmas, NULL);
    epsilon_j = 503.25f * strtod(epsilons, NULL); // convert from kcal/mol to K
    for (replica_x=-1; replica_x <=1; replica_x++)
    for (replica_y=-1; replica_y <=1; replica_y++)
    for (replica_z=-1; replica_z <=1; replica_z++)
    {
      configuration[number_of_atoms].x = strtod(xs, NULL) + replica_x * box_dimension;
      configuration[number_of_atoms].y = strtod(ys, NULL) + replica_y * box_dimension;
      configuration[number_of_atoms].z = strtod(zs, NULL) + replica_z * box_dimension;
      configuration[number_of_atoms].r_ij = pow(0.5f * (pow(r_i, 6) + pow(r_j, 6)), 0.1666666f);
      configuration[number_of_atoms].epsilon_ij = 2.0 * sqrt(epsilon_i * epsilon_j) * (pow(r_i, 3) * pow(r_j, 3)) / (pow(r_i, 6) + pow(r_j, 6));
      number_of_atoms++;
    }
  }
 
printf("atoms: %d\n", number_of_atoms);
  fclose(stdin);
}
