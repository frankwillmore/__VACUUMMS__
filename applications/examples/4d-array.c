#include <stdio.h>
#include <stdlib.h>

int ****ra;
int x_dim = 2;
int y_dim = 2;
int z_dim = 2;
int l_dim = 2;

main()
{
  int j,k,l;

  ra = (int****)malloc(sizeof(int***) * x_dim);
  for (j=0; j<y_dim; j++)
  {
    ra[j] = (int***)malloc(sizeof(int**) * y_dim);
    for (k=0; k<z_dim; k++) 
    {
      ra[j][k] = (int**)malloc(sizeof(int*) * z_dim); 
      for (l=0; l<l_dim; l++) 
      {
        ra[j][k][l] = (int*)malloc(sizeof(int) * l_dim);
      }
    }
  }

  ra[1][1][1][1] = 5;
  printf("%d\n", ra[1][1][1][1]);
}
