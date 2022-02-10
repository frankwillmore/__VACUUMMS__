#include <stdlib.h>
#include <stdio.h>

main(int argc, char *argv[])
{
  char data[65536];
  int i=0;
  int blocks = 65536;
  FILE *fout = stdout;
  FILE *fptr = fopen("/dev/urandom", "r");

  if ( strcmp(argv[1], "") ) blocks = strtol(argv[1], NULL, 10);
  printf("argv[1] = %d blocks to %s\n", blocks, argv[2]);

  if ( strcmp(argv[2], "") )  
  {
    fout = fopen(argv[2], "w");
    if (fout == NULL) 
    {
      printf("could not open %s\n", argv[2]);
      exit;
    }
  }
  for (i=0; i<blocks; i++)
  {
    fread(data, 65536, 1, fptr);
    fwrite(data, 65536, 1, fout);
  }
} 

