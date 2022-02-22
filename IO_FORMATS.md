# Applications User Guide

## IO formats

In general, VACUUMMS I/O uses non-binary representation of data. Human-readable text is sifted, stripped, sorted, etc. using POSIX pipes, records, and tab-separated columns. The most common formats are:

.cfg - 'configuration' or a simple set of x,y,z coordinates, ostensibly for a homogenous fluid, e.g. monodisperse hard-spheres or Lennard-Jones particles. E.g., for centers arranged in a FCC crystal:

    0.000000    0.000000    0.000000
    0.707107    0.707107    0.000000
    0.000000    0.707107    0.707107
    0.707107    0.000000    0.707107
    0.000000    0.000000    1.414214
    0.707107    0.707107    1.414214
    0.000000    0.707107    2.121320
    0.707107    0.000000    2.121320
    0.000000    0.000000    2.828427
    0.707107    0.707107    2.828427

.gfg - 'generalized configuration' extends the .cfg format by appending columns for Lennard-Jones sigma and epsilon values:

    0.000000    0.000000    0.000000    1.000   1.000
    0.707107    0.707107    0.000000    1.000   1.000
    0.000000    0.707107    0.707107    1.000   1.000
    0.707107    0.000000    0.707107    1.000   1.000
    0.000000    0.000000    1.414214    1.000   1.000
    0.707107    0.707107    1.414214    1.000   1.000
    0.000000    0.707107    2.121320    1.000   1.000
    0.707107    0.000000    2.121320    1.000   1.000
    0.000000    0.000000    2.828427    1.000   1.000
    0.707107    0.707107    2.828427    1.000   1.000

This five-column .gfg format is the most commonly used VACUUMMS input format and the starting point for working in VACUUMMS. Pre-processing your data to get it to this format will take some effort, but once rendered in this format, VACUUMMS is most elegant. 

.cav - Cavities generated using the CESA algorithm (implemented in ddx/pddx) as x,y,z,d (cavity diameter)
 
.cls - Clusters of cavities, with the first column containing the cluster number as primary key 

.fvi - Free volume index. A set of x,y,z coordinates plus the value of the Widom insertion probability at the specified point. This is intended to represent a (rather bulky) tabulated scalar field.

.tif[f] - 3-D TIFF standard format. FVI data is typically packed into a 3-D tiff for examination with a standard viewer.

.hst - Histogram, essentially a tabulated function. 

.vis - A primitive visualization format, to be consumed directly for crude realtime X11 output, but more broadly intended for conversion to .pov (ray-tracing format)

.pov - POVRay Scene Definition Language, components of which can be generated on a layer-by-layer basis before being composited to render an image. 

## Applications

These are used to either generate new or analyze existing molecular structures.

### ddx / pddx Cavity Energetic Sizing Algorithm (CESA)
Locates, then sizes cavities according to provided configuration and parameters. pddx is a thread-parallel version of ddx and requires pthreads to run. 

#### Options

    -box [ 6.0 6.0 6.0 ]           box dimensions 
    -seed [ 1 ]                    PRNG seed
    -randomize                     re-seed PRNG
    -characteristic_length [ 1.0 ] approximate length scale, e.g. typical atomic radius
    -characteristic_energy [ 1.0 ] approximate dimensionless energy scale
    -precision_parameter [ 0.001 ] delta value for precision of locating stationary points
    -n_steps [ 1000 ]              maximum number of steps to attempt before accepting convergence
                                   (roughly reciprocal of precision parameter)
    -show_steps                    list # of steps taken as final column
    -verlet_cutoff [ 100.0 ]       Square of radius used in determining inclusion in Verlet list
    -n [ 1 ]                       Number of samples to run
    -volume_sampling               Only use points whose trajectory originated within radius of final ghost particle
    -include_center_energy 
    -min_diameter [ 0.0 ]          Only include cavities whose diameter is at least this much.

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
