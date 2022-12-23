# Python interface to VACUUMMS

## vacuumms.py - python package

This file contains all wrappers and definitions. Place the directory containing this file in PYTHONPATH, and import vacuumms to use it, and/or import specific functions as needed. The VACUUMMS executables need to be in the PATH of the environment of the running python process. 

## Usage

Workflows built using the vacuumms command line leverage POSIX pipelines and use filters like awk and grep to prep data for the next pipeline stage, sometimes storing work in intermediate files. The python interface performs ultimately the same functionality, constructing objects from vacuumms datafiles and launching processes 

### Datatypes

Each of the vacuumms datatypes (gfg, cav, etc.) has a corresponding type of the same name in the vacuumms package. Data is stored as a list of records, e.g. `gfg.list`, and each record is composed of the fields used in that datatype. For example, a gfg data file contains tab-separated {x, y, z, sigma, epsilon} records and the python datatype contains records of the form `record.{x, y, z, sigma, epsilon}`. 

The constructor for these takes a filename from which to construct the datatype, plus the box dimensions as a tuple [[box_x], [box_y], [box_z]]:

`>>> fcc = vacuumms.gfg(input_file="fcc.gfg", box=[4.242640687119285, 4.242640687119285, 4.242640687119285])`

Since this is python, it's also possible to easily attach additional data (e.g. PVT, or other simulation-specific data) to this structure, either by ad-hoc declaration or by subclassing the datatype. 
 
### Operators

VACUUMMS operators take a VACUUMMS object as an input, plus a set of optional parameters, and return either a VACUUMMS object (in the case of a transforming operator) or a value (in the case of a reducing operator). Box dimensions are implicit, since they are recorded when constructing a VACUUMMS data object. For example, running the cavity sizing operator (ddx) on the object generated above:

`>>> my_cavs=vacuumms.ddx(fcc, n=5)`

### Output

Every vacuumms object provides a dump() method and a to_file() method which will, respectively, dump data to stdout, or to the filename specified. The following will dump the cavity set above to the file fcc.cav:

`>>> my_cavs.to_file("fcc.cav")`

### Other notes

usage
