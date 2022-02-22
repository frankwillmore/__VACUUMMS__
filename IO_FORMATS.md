# VACUUMMS IO formats

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

.gfgc - gfg plus one additional column specifying color (used for drawing the positive matter)

.cav - Cavities generated using the CESA algorithm (implemented in ddx/pddx) as x,y,z,d (cavity diameter)

.unq - same as cav, but implies the cavities are unique
 
.cls - Clusters of cavities, with the first column containing the cluster number as primary key 

.fvi - Free volume index. A set of x,y,z coordinates plus the value of the repulsive component of the Widom insertion probability (a.k.a. free volume index) at the specified point. This is intended to represent a (rather bulky) tabulated scalar field.

.tif[f] - 3-D TIFF standard format. FVI data is typically packed into a 3-D tiff for examination with a standard viewer.

.hst - Histogram, essentially a tabulated function. 

.vis - A primitive visualization format, to be consumed directly for crude realtime X11 output, but more broadly intended for conversion to .pov (ray-tracing format). Consists of x, y, z, diameter, and a number indicating a color (1 = Red, 2 = Green)

.pov - POVRay Scene Definition Language, components of which can be generated on a layer-by-layer basis before being composited to render an image. 

