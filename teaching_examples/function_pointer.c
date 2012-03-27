#include <stdio.h>

int function1()
{
  return 1;
}

main()
{
  int result;
  int (*pf)() = &function1;
  result = pf();
  printf ("%d\n", result);
} 
