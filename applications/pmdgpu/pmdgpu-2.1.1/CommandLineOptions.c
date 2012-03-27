/* CommandLineOptions.c */

#include <pmdgpu.h>

extern struct CommandLineOptions clo;
extern int mpi_rank;

void readCommandLineOptions(int argc, char* argv[])
{
  int i=0;

  clo.verbose = 0;
  clo.r_i = 2.9; // Angstroms
  clo.r_i1 = 2.9; // Angstroms
  clo.r_i2 = 2.9; // Angstroms
  clo.test_mass = 0.0440; // kg/mol
  clo.T = 298.0; // K
  clo.target_time = 100.0; // ps
  clo.verlet_cutoff_sq = 100.0; // Angstroms sq
  clo.seed = 123450;
  clo.box_x = 12.0;  // Angstroms
  clo.box_y = 12.0;  // Angstroms
  clo.box_z = 12.0;  // Angstroms
//  clo.n_insertions = 1;
  clo.n_insertions = 8;
  clo.n_threads = 8;
  clo.resolution_t = 10.0;
  clo.molecule = PMD_CO2;
  clo.config_directory = ".";
  clo.config_prefix = "";
  clo.config_suffix = ".gfg";
  clo.config_start = 0;
  clo.use_stdin = 0;
  clo.use_mpi = 0;
  clo.gross_resolution = 1.0f;
  clo.individual_trajectories = 0;

  while (++i<argc)
  {
    if (!strcmp(argv[i], "-usage")) 
    {
      printf("\nusage:  configuration in, list of path lengths  and times out\n\n");
      printf("        -verbose (startup parameters and information displayed at runtime) \n");
      printf("        -use_stdin (read configuration from standard input) \n");
      printf("        -use_mpi (run for multiple configs on multiple nodes) \n");
      printf("        -resolution_t [ 10.0 ] (interval between report times in picoseconds) \n");  
      printf("        -T [ 298.0 ] (absolute temperature in Kelvin)\n");
      printf("        -target_time [ 100.0 ] (how long to run the simulation in picoseconds)\n");
      printf("        -verlet_cutoff_sq (square of Verlet cutoff distance in Angstroms squared) [ 100.0 ]\n");
      printf("        -seed (seed value for random number generator) [ 123450 ]\n");
      printf("        -n [ 1 ] (number of test molecule insertions)\n");
      printf("        -N [ 8 ] (max number of concurrent threads)\n");
      printf("        -box [ 12.0 12.0 12.0 ] (Angstroms)\n");
      printf("        -config_directory [ . ] (where configurations are stored)\n");
      printf("        -config_prefix [ ] (e.g. HAB_ is prefix for HAB_1.gfg)\n");
      printf("        -config_suffix [ .gfg ] (file suffix)\n");
      printf("        -config_start [ 0 ] (which number configuration to begin with)\n");
      printf("        -individual_trajectories (list the path for each diffusant molecule)\n");
      printf("\n");
      exit(0);
    }
    else if (!strcmp(argv[i], "-verbose")) clo.verbose = 1;
    else if (!strcmp(argv[i], "-use_stdin")) clo.use_stdin = 1;
    else if (!strcmp(argv[i], "-use_mpi")) clo.use_mpi = 1;
    else if (!strcmp(argv[i], "-CO2")) clo.molecule = PMD_CO2;
    else if (!strcmp(argv[i], "-Oxygen")) clo.molecule = PMD_OXYGEN;
    else if (!strcmp(argv[i], "-Nitrogen")) clo.molecule = PMD_NITROGEN;
    else if (!strcmp(argv[i], "-Hydrogen")) clo.molecule = PMD_HYDROGEN;
    else if (!strcmp(argv[i], "-Helium")) clo.molecule = PMD_HELIUM;
    else if (!strcmp(argv[i], "-resolution_t")) clo.resolution_t = (strtod(argv[++i], NULL));
    else if (!strcmp(argv[i], "-test_mass")) clo.test_mass = (strtod(argv[++i], NULL));
    else if (!strcmp(argv[i], "-T")) clo.T = (strtod(argv[++i], NULL));
    else if (!strcmp(argv[i], "-target_time")) clo.target_time = (strtod(argv[++i], NULL));
    else if (!strcmp(argv[i], "-verlet_cutoff_sq")) clo.verlet_cutoff_sq = (strtod(argv[++i], NULL));
    else if (!strcmp(argv[i], "-threshold_time")) clo.threshold_time = (strtod(argv[++i], NULL));
    else if (!strcmp(argv[i], "-seed")) clo.seed = (strtol(argv[++i], NULL, 10));
    else if (!strcmp(argv[i], "-N")) clo.n_threads = (strtol(argv[++i], NULL, 10));
    else if (!strcmp(argv[i], "-n")) clo.n_insertions = (strtol(argv[++i], NULL, 10));
    else if (!strcmp(argv[i], "-config_directory")) clo.config_directory = argv[++i];
    else if (!strcmp(argv[i], "-config_prefix")) clo.config_prefix = argv[++i];
    else if (!strcmp(argv[i], "-config_suffix")) clo.config_suffix = argv[++i];
    else if (!strcmp(argv[i], "-config_start")) clo.config_start = (strtol(argv[++i], NULL, 10));
    else if (!strcmp(argv[i], "-individual_trajectories")) clo.individual_trajectories = 1;
    else if (!strcmp(argv[i], "-box"))
    {
      clo.box_x = (strtod(argv[++i], NULL));
      clo.box_y = (strtod(argv[++i], NULL));
      clo.box_z = (strtod(argv[++i], NULL));
    }
  }

  switch (clo.molecule)
  {
    case PMD_OXYGEN:  
      printf("Loading Oxygen parameters...\n");
      clo.bond_length1 = 0.6037; 
      clo.bond_length2 = -0.6037;
      clo.test_mass = 0.0f;
      clo.test_mass1 = 0.016f;
      clo.test_mass2 = 0.016f;
      clo.epsilon_i = 0.00;
    clo.epsilon_i = 0.1f;
      clo.epsilon_i1 = 39.23;
      clo.epsilon_i2 = 39.23;
      clo.r_i = 0.0f;
    clo.r_i = 0.1f;
      clo.r_i1 = 4.0070f;
      clo.r_i2 = 4.0070f;
      break;
    case PMD_NITROGEN:  
      printf("Loading Nitrogen parameters...\n");
      clo.bond_length1 = 0.54885; 
      clo.bond_length2 = -0.54885;
      clo.test_mass = 0.0f;
      clo.test_mass1 = 0.014f;
      clo.test_mass2 = 0.014f;
      clo.epsilon_i = 0.00;
    clo.epsilon_i = 0.1f;
      clo.epsilon_i1 = 30.09;
      clo.epsilon_i2 = 30.09;
      clo.r_i = 0.0f;
    clo.r_i = 0.1f;
      clo.r_i1 = 3.8008f;
      clo.r_i2 = 3.8008f;
      break;
    case PMD_HYDROGEN:  
      printf("Loading Hydrogen parameters...\n");
      clo.bond_length1 = 0.3706; 
      clo.bond_length2 = -0.3706;
      clo.test_mass = 0.0f;
      clo.test_mass1 = 0.00100f;
      clo.test_mass2 = 0.00100f;
      clo.epsilon_i = 0.00;
    clo.epsilon_i = 0.1f;
      clo.epsilon_i1 = 10.87;
      clo.epsilon_i2 = 10.87;
      clo.r_i = 0.0f;
    clo.r_i = 0.1f;
      clo.r_i1 = 1.4210f;
      clo.r_i2 = 1.4210f;
      break;
    case PMD_HELIUM:
      printf("Loading Helium parameters...\n\n");
      clo.test_mass = 0.004f;
      clo.r_i = 2.90;
      clo.epsilon_i = 2.50;
      clo.epsilon_i1 = 0.0;
      clo.epsilon_i2 = 0.0;
      break;
    case PMD_CO2:     
    default:
      printf("Loading CO2 parameters...\n");
      // CO2 code here
      clo.bond_length1 = 1.160; 
      clo.bond_length2 = -1.160;
      clo.test_mass = 0.012f;
      clo.test_mass1 = 0.016f;
      clo.test_mass2 = 0.016f;
      clo.epsilon_i = 34.22f;
      clo.epsilon_i1 = 33.72f;
      clo.epsilon_i2 = 33.72f;
      clo.r_i = 3.915;
      clo.r_i1 = 3.360;
      clo.r_i2 = 3.360;
      break;
  }

  if (mpi_rank==0 && clo.verbose==1) 
  {
    printf("\n\n\n");
    printf("verbose:  %d\n", clo.verbose);
    printf("test_diameter:  %f\n", clo.r_i);
    printf("test_diameter1:  %f\n", clo.r_i1);
    printf("test_diameter2:  %f\n", clo.r_i2);
    printf("test_epsilon:  %f\n", clo.epsilon_i);
    printf("test_epsilon1:  %f\n", clo.epsilon_i1);
    printf("test_epsilon2:  %f\n", clo.epsilon_i2);
    printf("test_mass:  %f\n", clo.test_mass);
    printf("test_mass1:  %f\n", clo.test_mass1);
    printf("test_mass2:  %f\n", clo.test_mass2);
    printf("T:  %f\n", clo.T);
    printf("target_time:  %f\n", clo.target_time);
    printf("resolution_t:  %f\n", clo.resolution_t);
    printf("verlet_cutoff_sq:  %f\n", clo.verlet_cutoff_sq);
    printf("threshold_time:  %f\n", clo.threshold_time); 
    printf("seed:  %d\n", clo.seed); 
    printf("N(number of threads):  %d\n", clo.n_threads);
    printf("n(number of insertions):  %d\n", clo.n_insertions);
    printf("config_directory:  %s\n", clo.config_directory);
    printf("config_prefix:  %s\n", clo.config_prefix);
    printf("config_suffix:  %s\n", clo.config_suffix);
    printf("config_start:  %d\n", clo.config_start);
    printf("box:  %f, %f, %f\n", clo.box_x, clo.box_y, clo.box_z);
    printf("\n\n\n");
  }

  fflush(stdout);

} /* end CommandLineOptions */

