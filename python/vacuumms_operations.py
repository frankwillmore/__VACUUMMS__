# This is vacuumms_operations.py, defines the operations 

import csv
import subprocess

from vacuumms_types import *

## VACUUMMS operations


def ddx(gfg, # box=[6.0, 6.0, 6.0],
             seed=1,
             randomize=False,
             characteristic_length=1.0,
             characteristic_energy=1.0,
             precision_parameter=0.001,
             n_steps=1000, 
             show_steps=False,
             verlet_cutoff=100.0,
             n=1,
             volume_sampling=False,
             include_center_energy=False,
             min_diameter=0.0):

    ddx_arglist = ['ddx', '-box', str(gfg.box[0]), str(gfg.box[1]), str(gfg.box[2]) ]
    ddx_arglist.append("-n_steps")
    ddx_arglist.append(str(n_steps))
    ddx_arglist.append("-seed")
    ddx_arglist.append(str(seed))
    if (randomize): ddx_arglist.append("-randomize")
    ddx_arglist.append("-characteristic_length")
    ddx_arglist.append(str(characteristic_length))
    ddx_arglist.append("-characteristic_energy")
    ddx_arglist.append(str(characteristic_energy))
    ddx_arglist.append("-precision_parameter")
    ddx_arglist.append(str(precision_parameter))
    ddx_arglist.append("-n_steps")
    ddx_arglist.append(str(n_steps))
    if (show_steps): ddx_arglist.append("-show_steps")
    ddx_arglist.append("-verlet_cutoff")
    ddx_arglist.append(str(verlet_cutoff))
    ddx_arglist.append("-n")
    ddx_arglist.append(str(n))
    if (volume_sampling): ddx_arglist.append("-volume_sampling")
    if (include_center_energy): ddx_arglist.append("-include_center_energy")
    ddx_arglist.append("-min_diameter")
    ddx_arglist.append(str(min_diameter))

    tab = str("\t")
    newline = str("\n")

    # open the process for ddx, and write gfg info
    with subprocess.Popen(ddx_arglist, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as process:
        print("FTW subprocess for ddx: " + str(process))
        for row in gfg.gfg_list:
            line = str(row.x)+tab+str(row.y)+tab+str(row.z)+tab+str(row.sigma)+tab+str(row.epsilon)+newline
            process.stdin.write(bytes(line, encoding='utf-8'))

        # Done writing input data so close stdin
        process.stdin.close()

        # Capture the output stream to create cav object
        ddx_retval = cav(process.stdout, gfg.box)

        process.stdout.close()

    return ddx_retval

