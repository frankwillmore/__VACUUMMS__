# User Guide

General notes:

These are the notes.

## Applications

These are used to either generate new or analyze existing molecular structures.

### ddx / pddx Cavity Energetic Sizing Algorithm (CESA)
Locates, then sizes cavities according to provided configuration and parameters. pddx is a thread-parallel version of ddx and requires pthreads to run. 

#### Options

    -box [ 6.0 6.0 6.0 ]            
    -seed [ 1 ]
    -randomize 
    -characteristic_length [ 1.0 ]
    -characteristic_energy [ 1.0 ]
    -precision_parameter [ 0.001 ]
    -n_steps [ 1000 ] (roughly reciprocal of precision parameter)
    -show_steps (includes steps taken as final column)
    -verlet_cutoff [ 100.0 ]
    -n [ 1 ]
    -volume_sampling 
    -include_center_energy 
    -min_diameter [ 0.0 ]

### ljx
Lennard-Jones fluid.

#### Options

  [-v]                            verbose
  [-ng]                           no X11 graphics
  [-no_side]                      don't show side
  -T [1]                          temperature
  -fixed_perturbation_length []   
  -particle_scale []
  -N                              number of particles
  -relaxation_allowance
  -box                            box dimensions (triplet)
  -fg_color [] 
  -bg_color [] 
  -min_color
  -rng_seed []                    initialize random number generator with given seed
  -randomize                      initialize random number generator with random seed
  -end_mcs []                     how many Monte Carlo steps to run
  -energy_report_frequency []
  -configuration_threshold []
  -configuration_frequency [] 
  -input_file_name                load configuration from a file
  -target_acceptance_ratio []     [0..1) fraction of MC moves to accept
  -psi_shift                      truncated and shifted potential, shift value
  -cutoff_sq                      square of Verlet cutoff radius

### hs

#### Options

## Libraries

The libraries here typically used to provide functionality which has been factored from the provided executables.

## Utils

The utils included are typically quick text-processing utilities that perform simple manipulations such as smoothing data, sorting/histogramming, translating one format to another, etc.
