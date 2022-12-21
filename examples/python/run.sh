#!/usr/bin/env python

import vacuumms

# create a gfg object from a file
fcc = vacuumms.gfg(input_file="fcc.gfg", dims=[4.242640687119285, 4.242640687119285, 4.242640687119285])

# Check what was read
#fcc.dump()

# start a process to run ddx, and return .cav object
my_cavs=vacuumms.ddx(fcc)

# Dump the result
my_cavs.dump()


