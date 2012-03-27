#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[])
{
  char line[80];
  double x;
  double x_sum;
  int i;

  while (1)
  {
    fgets(line, 80, stdin);
    if (feof(stdin)) break;

    x = strtod(line, NULL);

    x_sum += x;
  }

  printf("%lf\n", x_sum);
}
