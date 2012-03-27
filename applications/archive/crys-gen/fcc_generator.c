/* crystal_generator.c */

#include <stdio.h>

int main(int argc, char *argv[])
{
  double l = 1.414213562373;
  double i, j, k;
  double max_x = 6;
  double max_y = 6;
  double max_z = 6;

  for (i=0; i<3; i++)
  {
    for (j=0; j<3; j++) 
    {
      for (k=0; k<3; k++)
      {
        printf("%lf\t%lf\t%lf\n", i*l, j*l, k*l);
        printf("%lf\t%lf\t%lf\n", i*l + l/2, j*l + l/2, k*l);
        printf("%lf\t%lf\t%lf\n", i*l, j*l + l/2, k*l + l/2);
        printf("%lf\t%lf\t%lf\n", i*l + l/2, j*l, k*l + l/2);
      }
    }
  }

  return 0;
}
