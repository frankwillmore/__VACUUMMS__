/* ck.c */

#include <ftw_std.h>
#include <ftw_rng2.h>
#include <ftw_param.h>
#include <errno.h>

// In:  .cav 

int main(int argc, char* argv[])
{
  double x, y, z, d;
  char line[80];
  char *xs, *ys, *zs, *ds;
  int lines_read=0;

  setCommandLineParameters(argc, argv);

  while (TRUE)
  {
    fgets(line, 80, stdin);
    if (feof(stdin)) break;

    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");
    ds = strtok(NULL, "\n");

    errno = 0;

    x = strtod(xs, NULL);
    y = strtod(ys, NULL);
    z = strtod(zs, NULL);
    d = strtod(ds, NULL);

    printf("%d\n", ++lines_read);
    fflush(stdout);
//    printf("%lf\t%lf\t%lf\t%lf\t%lf\n", x, y, z, sigma, epsilon);
  }
}

