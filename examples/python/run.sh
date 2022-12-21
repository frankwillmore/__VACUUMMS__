#!/usr/bin/env python

import vacuumms

#f=vacuumms.v_open_file("xyz.dat")
#vacuumms.v_drain_pipe(pipe=f, file="xyz.out")
#outfile=open("file.out", "a")

# create a gfg object from a file
fcc = vacuumms.gfg(input_file="fcc.gfg", dims=[4.242640687119285, 4.242640687119285, 4.242640687119285])

# check what was read
#fcc.dump()

# start a process to run pddx, and return .cav object
my_cavs=vacuumms.pddx(fcc)

print("FTW got retval:")
print(my_cavs)




