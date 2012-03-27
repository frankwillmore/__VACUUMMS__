/* clep.c */

/* Cavity life expectancy protocol */

#include <stdio.h>
#include <math.h>

#include <ftw_param.h>
#include <ftw_types.h>
#include <ftw_cav_parser.h>
#include <ftw_config_parser.h>
#include <ftw_gfgEnergy.h>

#define max_cavs 100000

double box_x, box_y, box_z;

char *input_file_name;
char *config_directory;
char config_path[256];

int extinction_time[max_cavs]; // Initialized to zero, so can also be used as a truth test
//ftw_Cavity cavities[max_cavs]; 

int main(int argc, char *argv[])
{
  int cav;

  setCommandLineParameters(argc, argv);

  if (getFlagParam("-usage")) {  
    printf("usage:	clep	\n");
    printf("			-box [n.nn] [n.nn] [n.nn] \n");
    printf("			-input_file (.cav format) \n");
    printf("			-config_directory (0000.gfg, 0001.gfg, ... ) \n");
    printf("\n");
    exit(0);
  }

  if (!getFlagParam("-input_file")) 	{	printf("no input file\n");		exit(0);	}
  if (!getFlagParam("-config_directory")) {	printf("no config directory\n");	exit(0);	}

  // read cavity file
  getStringParam("-input_file", &input_file_name);
  FILE *instream = fopen(input_file_name, "r");
  ftw_CAV65536 *cavities = readCAV65536(instream);

  getVectorParam("-box", &box_x, &box_y, &box_z);
  cavities->box_x = (float)box_x;
  cavities->box_y = (float)box_y;
  cavities->box_z = (float)box_z;

  // loop through gfg files
  FILE *ls;
  char buffer[256];
  char *gfg;
  char ls_command[256];
  getStringParam("-config_directory", &config_directory);
  sprintf(ls_command, "ls %s", config_directory);

  ls = popen(ls_command, "r"); 
  int config_number=0;
  while (fgets(buffer, 256, ls) != NULL)
  {
    gfg = strtok(buffer, "\n");
    config_number++;
    sprintf(config_path, "%s/%s", config_directory, gfg);
//fprintf(stderr, "config: %s\n", config_path);
    FILE *config_file = fopen(config_path, "r");
ftw_GFG65536 *_p_gfg = readGFG65536(config_file);
    _p_gfg->box_x = (float)box_x;
    _p_gfg->box_y = (float)box_y;
    _p_gfg->box_z = (float)box_z;
    ftw_GFG65536 *p_gfg = replicateGFG65536(_p_gfg);


// p_gfg = _p_gfg;
    for (cav=0; cav < cavities->n_cavities; cav++)
    {
      // calc energy of this cav in thiss config
      // mark extinction as they are discovere
      if ( !extinction_time[cav])
      {
        //float energy = ftw_GFG65536Energy_612(p_gfg, cavities->cavity[cav].x, cavities->cavity[cav].y, cavities->cavity[cav].z, 0.0f);
        //if (energy >100.0f) 
	if (ftw_GFG65536HS_Overlap(p_gfg, cavities->cavity[cav].x, cavities->cavity[cav].y, cavities->cavity[cav].y, 0.0f))
        {
//printf("boom! cav %d at %d with energy %f\n", cav, config_number, energy);
          extinction_time[cav] = config_number;
        }
      }
    } 

    // And clean up the pieces... no leaky!
    free(_p_gfg);
    free(p_gfg);
  } 

  if (pclose(ls)) {	printf("error closing child process.\n"); exit(1);	}

  for (cav=0; cav<cavities->n_cavities; cav++) printf("%d\t%d\t%f\n", cav, extinction_time[cav], cavities->cavity[cav].diameter);

  return 0; // main() normal exit
}
