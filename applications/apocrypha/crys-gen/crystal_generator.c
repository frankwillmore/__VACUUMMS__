/* crystal_generator.c */

#include <stdio.h>

int main(int argc, char *argv[])
{
  double step = 3.0;
  double x, y, z;
  double max_x = 6;
  double max_y = 6;
  double max_z = 6;

  for (x=1; x<max_x; x+=step)
  {
    for (y=1; y<max_y; y+=step) 
    {
      for (z=1; z<max_z; z+=step)
      {
        printf("%lf\t%lf\t%lf\n", x, y, z);
      }
    }
  }

  return 0;
}
