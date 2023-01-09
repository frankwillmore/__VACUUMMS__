#!/usr/bin/env python

import vacuumms

# create a gfg object from a file
fcc = vacuumms.gfg(input_file="fcc.gfg", box=[4.242640687119285, 4.242640687119285, 4.242640687119285])

# run ddx, and return a cav object
my_cavs=vacuumms.ddx(fcc, n=5)

# display the result, with metadata
# my_cavs.dump()

# write result to file
my_cavs.to_file("fcc.cav")


