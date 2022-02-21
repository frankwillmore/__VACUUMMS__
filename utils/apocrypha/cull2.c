/* cull.c */

#include <stdio.h>
#include <string.h>
#include <ftw_std.h>

short int use_point[111][111][111];

double box_x=10, box_y=10, box_z=10;
int number_of_cavities=0;

double resolution=0.1;
double test_diameter=1.0;
int n_value=0;

int max_x_value;
int max_y_value;
int max_z_value;

void readInputStream();

int main(int argc, char *argv[])
{
  int x_minus_ok, x_plus_ok;
  int y_minus_ok, y_plus_ok;
  int z_minus_ok, z_plus_ok;

  int i, j, k, ri, rj, rk;

  setCommandLineParameters(argc, argv);
  getVectorParam("-box", &box_x, &box_y, &box_z);
  getDoubleParam("-resolution", &resolution);
  getDoubleParam("-test_diameter", &test_diameter);
  if (getFlagParam("-usage"))
  {
    printf("\n");
    printf("usage:\t-box [ 10 10 10 ]\n");
    printf("      \t-resolution [ .10 ]\n");
    printf("      \t-test_diameter [ 1.0 ]\n");
    printf("\n");
    exit(0);
  }

  n_value = test_diameter/(resolution*sqrt(3));
  max_x_value = floor(.000001 + box_x/resolution) - n_value;
  max_y_value = floor(.000001 + box_y/resolution) - n_value;
  max_z_value = floor(.000001 + box_z/resolution) - n_value;
 
  readInputStream(); 

  // now cull the useful vals...

 
  for (i=n_value; i<max_x_value; i++)
  for (j=n_value; j<max_y_value; j++)
  for (k=n_value; k<max_z_value; k++)
  {
    if (use_point[i][j][k]) 
    {
      x_minus_ok=0;
      z_minus_ok=0;
      y_minus_ok=0;
      x_plus_ok=0;
      y_plus_ok=0;
      z_plus_ok=0;

      for (ri=i-n_value; ri < i; ri++)
      if (use_point[ri][j][k])
      {
        x_minus_ok = 1;
        break;
      }
      for (ri=i+1; ri <= i + n_value; ri++)
      if (use_point[ri][j][k])
      {
        x_plus_ok = 1;
        break;
      }

      for (rj=j-n_value; rj < j; rj++)
      if (use_point[i][rj][k]) 
      {
        y_minus_ok = 1;
        break;
      }
      for (rj=j+1; rj <= j + n_value; rj++)
      if (use_point[i][rj][k])
      {
        y_plus_ok = 1;
        break;
      }

      for (rk=k-n_value; rk < k; rk++)
      if (use_point[i][j][rk]) 
      {
        z_minus_ok = 1;
        break;
      }
      for (rk=k+1; rk <= k + n_value; rk++)
      if (use_point[i][j][rk])
      {
        z_plus_ok = 1;
        break;
      }

    if (x_minus_ok && x_plus_ok && y_minus_ok && y_plus_ok && z_minus_ok && z_plus_ok) 
      use_point[i][j][k] = 0;
    }
  }

  for (i=0; i<floor(.000001 + box_x/resolution); i++)
  for (j=0; j<floor(.000001 + box_y/resolution); j++)
  for (k=0; k<floor(.000001 + box_z/resolution); k++)
  if (use_point[i][j][k]) 
    printf("%lf\t%lf\t%lf\t%lf\n", i*resolution, j*resolution, k*resolution, test_diameter);
}

void readInputStream()
{
  char line[80];
  char *xs, *ys, *zs, *ds;
  double x, y, z, d;
  int x_int, y_int, z_int;
  int i, j, k;
int lines_read=0;

  for (i=0; i<100; i++)
    for (j=0; j<100; j++)
      for (k=0; k<100; k++)
        use_point[i][j][k] = 0;

  while (TRUE)
  {
    fgets(line, 80, stdin);
    if (feof(stdin)) break;

    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");
    ds = strtok(NULL, "\n");

    x = strtod(xs, NULL);
    y = strtod(ys, NULL);
    z = strtod(zs, NULL);
    d = strtod(ds, NULL);

    x_int = floor(x/resolution + .000001);
    y_int = floor(y/resolution + .000001);
    z_int = floor(z/resolution + .000001);
    use_point[x_int][y_int][z_int] = 1;
//printf("%d\n", lines_read++);
  }
}
