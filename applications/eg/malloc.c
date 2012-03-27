#include <stdlib.h>

typedef struct complex_double
{
  double a;
  double b;
} complex_number;

//typedef struct complex_double complex_number;

main(int argc, char* argv[])
{
  complex_number *z_ptr;
  complex_number z;
  z_ptr = malloc(sizeof(complex_number));
  
  z = *z_ptr;

  z.a = 1.5;
  z.b = 1.4;

  printf("%lf, %lf\n", z.a, z.b);
}
