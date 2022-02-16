/*************************************** vis.c ********************************************/

#include "vis.h"
#include "io_setup.h"
#include "graphics.h"
#include "command_line_parser.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <ftw_science.h>
#include <ftw_std.h>

#ifndef MAX_NUMBER_MOLECULES
#define MAX_NUMBER_MOLECULES 16384
#endif

/* non-configurable global params */
double x[MAX_NUMBER_MOLECULES], y[MAX_NUMBER_MOLECULES], z[MAX_NUMBER_MOLECULES], d[MAX_NUMBER_MOLECULES];
int c[MAX_NUMBER_MOLECULES];

int wsize_x, wsize_y, wsize_z;

int number_of_molecules;

int main(int argc, char *argv[])
{
  parseCommandLineOptions(argc, argv);
printf("%d\n",   loadConfiguration());
  initializeDisplay();

  while (1)
  {
    drawGraphicalRepresentation();
    checkForWindowEvent();
    sleep(5);
  }

  return 0;

} /* end main */
