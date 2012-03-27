/* miss.c */

/* input:  .dst */
/* output:  singular value (average of all entries) */

#include <ftw_std.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define N_LINES 100000

int line_indices[N_LINES];

int main(int argc, char *argv[])
{
  char line[80];
  char *indexs;
  int line_index, line_count=0;
  int low_index=0, high_index=0;
  int i, index;

  for (i=0; i<N_LINES; i++) line_indices[i] = 0;

  // get low index from first line

  fgets(line, 80, stdin);
  indexs = strtok(line, "\t");
  low_index = strtol(indexs, NULL, 10);
  line_count++;
  line_indices[low_index] = 1;

  while (TRUE)
  {
    fgets(line, 80, stdin);
    if (feof(stdin)) break;

    indexs = strtok(line, "\t");
    index = strtol(indexs, NULL, 10);
 
    if (index>high_index) high_index = index;
    if (index<low_index) low_index = index;

    line_count++;
    line_indices[index] = 1;
  }

  for (i=low_index; i<=high_index; i++) if (line_indices[i] == 0) printf("%d\n", i);
}
