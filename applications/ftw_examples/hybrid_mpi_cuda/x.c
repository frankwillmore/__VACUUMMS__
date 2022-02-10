#include <stdio.h>
#include <stdlib.h>

main(int argc, char* argv[])
{
  int i;
  char *ftw = getenv("_FTW");
 
  printf("_FTW=%s\n", ftw);
  ftw[2] = ' ';
  printf("_FTW=%s\n", ftw);
  ftw = getenv("_FTW");
  printf("_FTW=%s\n", ftw);

  for (i=0; i<argc; i++) printf("argv[%d] = %s\n", i, argv[i]);
}
