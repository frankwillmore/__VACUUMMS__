/* io_setup.c */

#include <ftw_std.h>
#include <ftw_rng.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "io_setup.h"
#include "hs_main.h"

//extern char simulation_unique_identifier[32];
//extern char *simulation_unique_identifier;
extern char *p_simulation_unique_identifier;
extern double temperature;
extern int number_of_molecules;
extern int energy_report_frequency;
extern int configuration_threshold;
extern int configuration_frequency;
extern double box_x, box_y, box_z;
extern int monte_carlo_steps;
extern int end_mcs;
extern char* log_file_name;
extern char* output_file_name;
extern char* input_file_name;
extern double x[], y[], z[];

double initial_spacing = 1.03;

FILE *log_file;
FILE *output_file;
FILE *input_file;

char hostname[50];
char *log_path;
char *results_path;

time_t now;

extern int verbose;

void setFCCInitialCondition()
{
  double root2 = 1.414213562373;
  double xx, yy, zz;
  int n=0;
  double lattice_param = 2*initial_spacing/root2;

  for (xx=0; xx<box_x-lattice_param; xx += lattice_param)
  for (yy=0; yy<box_y-lattice_param; yy += lattice_param)
  for (zz=0; zz<box_z-lattice_param; zz += lattice_param)
  {
    if (n<number_of_molecules)
    {
      x[n] = xx;
      y[n] = yy;
      z[n] = zz;
      n++;
    }
    if (n<number_of_molecules)
    {
      x[n] = xx;
      y[n] = yy + lattice_param/2;
      z[n] = zz + lattice_param/2;
      n++;
    }
    if (n<number_of_molecules)
    {
      x[n] = xx + lattice_param/2;
      y[n] = yy;
      z[n] = zz + lattice_param/2;
      n++;
    }
    if (n<number_of_molecules)
    {
      x[n] = xx + lattice_param/2;
      y[n] = yy + lattice_param/2;
      z[n] = zz;
      n++;
    }
  }

  if (n<number_of_molecules) 
  {
    printf("too many molecules... only %d/%d placed.\n", n, number_of_molecules);
    exit(1);
  }
}

void XXsetInitialConditions()
{
  int target_number_of_molecules;

  if (input_file_name != NULL) loadConfiguration();
  else 
  {
    monte_carlo_steps = 0;
    target_number_of_molecules = number_of_molecules;
    for (number_of_molecules=0; number_of_molecules<target_number_of_molecules; number_of_molecules++)
    {
      while (1)
      {
        x[number_of_molecules] = rnd() * box_x;
        y[number_of_molecules] = rnd() * box_y;
        z[number_of_molecules] = rnd() * box_z;
      
        if (!checkForOverlap(number_of_molecules)) break;
      }
printf("added #%d\n", number_of_molecules);
    }
  }
}

void setInitialConditions()
{
  //int xx, yy, zz;
  //int num_so_far=0;

  if (input_file_name != NULL) loadConfiguration();
  else setFCCInitialCondition();
/*
  {
    monte_carlo_steps = 0;
    for (xx=0; xx<box_x; xx+=1.1)
    for (j=0; j<box_y; j++)
    for (k=0; k<box_z; k++)
    {
      x[num_so_far] = i*1.1;   
      y[num_so_far] = j*1.1;   
      z[num_so_far] = k*1.1;   

      if (num_so_far++ >= number_of_molecules) return;
    }

    printf ("too many molecules...\n");
    exit(0);
  }
*/
}

void loadConfiguration()
{
  FILE *datastream;
  char line[80];
  char *xs, *ys, *zs;

  number_of_molecules = 0;
  V printf("loading %s...\n", input_file_name);
  datastream = fopen(input_file_name, "r");

  while (1)
  {
    fgets(line, 80, datastream);
    if (feof(datastream)) break;

    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\n");

    x[number_of_molecules] = strtod(xs, NULL);
    y[number_of_molecules] = strtod(ys, NULL);
    z[number_of_molecules++] = strtod(zs, NULL);
  }
 
  V printf("%d lines read.\n", number_of_molecules);
  fclose(datastream);
}

void generateUniqueId()
{
  int i;
  for (i=0; i<16; i++) *(p_simulation_unique_identifier + i) = (char)(rnd() * 26 + 65);
}

void readEnvironmentVariables()
{
  gethostname(hostname, 50);
  log_path = getenv("LOG_PATH");
  if (log_path != NULL) 
  {
    log_file_name = strcat(log_path, "/");
    log_file_name = strcat(log_file_name, hostname);
    log_file_name = strcat(log_file_name, "-hs.log");
  }
  results_path = getenv("RESULTS_PATH");
  if (results_path != NULL) 
  {
    output_file_name = strcat(results_path, "/");
    output_file_name = strcat(output_file_name, p_simulation_unique_identifier);
    output_file_name = strcat(output_file_name, "-hs.out");
  }
}

/* display headers on output */
void initializeOutput()
{
  now = time(NULL);

  if (verbose)
  {
    printf("#HT simulation %s started on %s:  %s", p_simulation_unique_identifier, hostname, ctime(&now));
    printf("#HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH\n");
    printf("#H N=%d\n", number_of_molecules);
    printf("#H box dimensions:  \t%lf x %lf x %lf = %lf\n", box_x, box_y, box_z, box_x*box_y*box_z);
    printf("#H reduced density = %lf\n", (number_of_molecules / (box_x * box_y * box_z)));
    printf("#H\n");
    printf("#H configuration threshold:\t%d\n", configuration_threshold);
    printf("#H configuration frequency:\t%d\n", configuration_frequency);
    printf("#H run until mcs = %d\n", end_mcs);
    printf("#HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH\n");
    printf("#H\n");
    printf("#H\n");
  }

  log_file = fopen(log_file_name, "a");
  fprintf(log_file, "simulation %s launched on %s:  %s", p_simulation_unique_identifier, hostname, ctime(&now));
  fclose(log_file);

  output_file = fopen(output_file_name, "w");
  fprintf(output_file, "#HT simulation %s started on %s:  %s", p_simulation_unique_identifier, hostname, ctime(&now));
  fprintf(output_file, "#HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH\n");
  fprintf(output_file, "#H N=%d\n", number_of_molecules);
  fprintf(output_file, "#H box dimensions:  \t%lf x %lf x %lf = %lf\n", box_x, box_y, box_z, box_x*box_y*box_z);
  fprintf(output_file, "#H reduced density = %lf\n", (number_of_molecules / (box_x * box_y * box_z)));
  fprintf(output_file, "#H\n");
  fprintf(output_file, "#H configuration threshold:\t%d\n", configuration_threshold);
  fprintf(output_file, "#H configuration frequency:\t%d\n", configuration_frequency);
  fprintf(output_file, "#H run until mcs = %d\n", end_mcs);
  fprintf(output_file, "#HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH\n");
  fprintf(output_file, "#H\n");
  fprintf(output_file, "#H\n");
  fflush(output_file);
}

void finalizeOutput()
{
  now = time(NULL);
  if (verbose) printf("#HT simulation %s finished on %s:  %s", p_simulation_unique_identifier, hostname, ctime(&now));
  fprintf(output_file, "#HT simulation %s finished on %s:  %s", p_simulation_unique_identifier, hostname, ctime(&now));
  fclose(output_file);

  log_file = fopen(log_file_name, "a");
  fprintf(log_file, "simulation %s finished on %s:  %s", p_simulation_unique_identifier, hostname, ctime(&now));
  fclose(log_file);
}

void generateOutput()
{
  int i;

  if ((monte_carlo_steps > configuration_threshold) && (monte_carlo_steps % configuration_frequency == 0))
  {
    now = time(NULL);
    log_file = fopen(log_file_name, "a");
    fprintf(log_file, "dumping configuration for %s on %s at %d steps:  %s", hostname, \
            p_simulation_unique_identifier, monte_carlo_steps, ctime(&now));
    fclose(log_file);

    if (verbose)
    {
      printf("#HC%06d\n", monte_carlo_steps);
      printf("#HC%06d dumping configuration at mcs=%d...\n", monte_carlo_steps, monte_carlo_steps);
      printf("#HC%06d\n", monte_carlo_steps);
      for (i=0; i<number_of_molecules; i++) printf("#C%06d\t%d\t%lf\t%lf\t%lf\n", monte_carlo_steps, i, x[i], y[i], z[i]);
      printf("#HC%06d\n", monte_carlo_steps);
    }

    fprintf(output_file, "#HC%06d\n", monte_carlo_steps);
    fprintf(output_file, "#HC%06d dumping configuration at mcs=%d...\n", monte_carlo_steps, monte_carlo_steps);
    fprintf(output_file, "#HC%06d\n", monte_carlo_steps);
    for (i=0; i<number_of_molecules; i++) fprintf(output_file, "#C%06d\t%d\t%lf\t%lf\t%lf\n", monte_carlo_steps, i, x[i], y[i], z[i]);
    fprintf(output_file, "#HC%06d\n", monte_carlo_steps);
  }

  fflush(output_file);
}

