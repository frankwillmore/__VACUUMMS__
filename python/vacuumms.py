# This is vacuumms.py, defines the package 

import csv
import subprocess


## VACUUMMS types 


# superclass for all data objects

class VACUUMMS:

    def __init__(self, dims):
        print("constructing VACUUMMS object")
        self.dims = dims

    def dump(self):
        print("box_x=" + str(self.dims[0])) 
        print("box_y=" + str(self.dims[1])) 
        print("box_z=" + str(self.dims[2])) 


class cav_record():
    def __init__(self, record):
        (self.x, self.y, self.z, self.d) = record

class cav(VACUUMMS):

    def __init__(self, input_source, dims):
        super().__init__(dims)
        print("constructing VACUUMMS::cav object")
        self.cav_list = []

        # Capture the output stream instead of a file to create the cav object

        if (str(type(input_source)) == "<class '_io.BufferedReader'>"):
            for line in input_source:
                self.cav_list.append(cav_record(line.decode().split("\t")))

        else: # otherwise treat input_source as a file
            with open(input_file, "r", encoding="utf8") as cav_file:
                reader = csv.reader(cav_file, delimiter="\t")
                for row in reader:
                    self.cav_list.append(cav_record(row))

    def dump(self):
        super().dump() # Dump PVT, dims, etc
        for record in self.cav_list:
            print(f"{record.x}\t{record.y}\t{record.z}\t{record.d}")


class gfg_record():
    def __init__(self, record):
        (self.x, self.y, self.z, self.sigma, self.epsilon) = record

class gfg(VACUUMMS):
    def __init__(self, input_file, dims):
        super().__init__(dims)
        print("constructing VACUUMMS::gfg object from "+input_file + " with dims = " + str(dims))
        self.gfg_list = []

        with open(input_file, "r", encoding="utf8") as gfg_file:
            reader = csv.reader(gfg_file, delimiter="\t")
            for row in reader:
                self.gfg_list.append(gfg_record(row))

    def dump(self):
        super().dump() # Dump PVT, dims, etc
        for record in self.gfg_list:
            print(f"{record.x}\t{record.y}\t{record.z}\t{record.sigma}\t{record.epsilon}")


## VACUUMMS methods


def ddx(gfg, n_steps=1000000, verlet_cutoff=100.0, n=1, min_diameter=0.0):

    ddx_arglist = ['ddx', '-box', str(gfg.dims[0]), str(gfg.dims[1]), str(gfg.dims[2]) ]
    ddx_arglist.append("-n_steps")
    ddx_arglist.append(str(n_steps))
    print("FTW using ddx_arglist: " + str(ddx_arglist))

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
        ddx_retval = cav(process.stdout, gfg.dims)

        process.stdout.close()

    return ddx_retval

