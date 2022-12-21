# This is vacuumms.py, defines the package 

import csv
import subprocess

def file_eater(file="x"):
    print("eating file " + file)

# Need to make a vacuumms object subclass, that has input and output streams, and an action

def v_open_file(file="/dev/null"):
    retval = open(file)
    return retval
    
    # Open a file and return a pipe


def v_drain_pipe(pipe="", file="/dev/null"):
    # Drain a pipe to a file ?
    f=open("file", "w")
    for line in pipe.stdout:
        f.write(line)
#        print(line.rstrip().decode('ASCII'))

# open the file in dst2hst. Can fix stderr later. 
def dst2hst():
    f=open('x.dst', "r")
    d2h=subprocess.Popen("dst2hst", stdout=subprocess.PIPE, stdin=f)

# Just dump pipe to stdout
#    for line in d2h.stdout:
#        print(line.rstrip().decode('ASCII'))

    
# superclass for all data objects
class VACUUMMS:
    def __init__(self, input_file, dims):
        print("constructing VACUUMMS object")
        self.box_x=dims[0]
        self.box_y=dims[1]
        self.box_z=dims[2]

    def dump(self):
        print("box_x=" + str(self.box_x)) 
        print("box_y=" + str(self.box_y)) 
        print("box_z=" + str(self.box_z)) 
    

class cav(VACUUMMS):
    def __init__(self, input_file, dims):
        super().__init__(input_file, dims)
        print("constructing VACUUMMS::cav object")

        with open(input_file, "r", encoding="utf8") as cav_file:
            reader = csv.reader(cav_file, delimiter="\t")
            for row in reader:
                (x, y, z, d) = row
                print(f"x={x} y={y} z={z} d={d}")
        
        print("Read " + str(input_file))


class gfg(VACUUMMS):
    def __init__(self, input_file, dims):
        super().__init__(input_file, dims)
        print("constructing VACUUMMS::gfg object from "+input_file + " with dims = " + str(dims))
        self.dims = dims
        self.gfg_list = []

        with open(input_file, "r", encoding="utf8") as gfg_file:
            reader = csv.reader(gfg_file, delimiter="\t")
            for row in reader:
                (x, y, z, sigma, epsilon) = row
                self.gfg_list.append(row);

    def dump(self):
#        print("Dumping gfg_list")
#        print(self.gfg_list)
        for row in self.gfg_list:
            (x, y, z, sigma, epsilon) = row
            print(f"x={x} y={y} z={z} sigma={sigma} epsilon={epsilon}")

#pddx usage:	-box [ 6.0 6.0 6.0 ]
#		-seed [ 1 ]
#		-characteristic_length [ 1.0 ] if in doubt, use largest sigma/diameter
#		-characteristic_energy [ 1.0 ] if in doubt, use largest energy/epsilon
#		-n_steps [ 1000000 ] maximum before giving up
#		-n_threads [ 1 ] 
#		-show_steps (includes steps taken as final column)
#		-verlet_cutoff [ 100.0 ]
#		-n_samples [ 1 ]
#		-volume_sampling 
#		-include_center_energy 
#		-min_diameter [ 0.0 ] 0.0 will give all 

def pddx(gfg, n_steps=1000000, n_threads=1, verlet_cutoff=100.0, n_samples=1, min_diameter=0.0):
#    f=open('x.dst', "r")
#    d2h=subprocess.Popen("dst2hst", stdout=subprocess.PIPE, stdin=f)
#    wc = subprocess.Popen(['wc', '-l'], stdin=subprocess.PIPE)

    pddx_retval = []
#    pddx_arglist = ['pddx', '-box ', str(gfg.dims[0]), str(gfg.dims[1]), str(gfg.dims[2]) ]
    pddx_arglist = ['ddx', '-box ', str(gfg.dims[0]), str(gfg.dims[1]), str(gfg.dims[2]) ]
    pddx_arglist.append("-n_steps " + str(n_steps))
    print("FTW pddx_arglist: " + str(pddx_arglist))

    # open the process for pddx, write info, and capture output
    # cav = subprocess.Popen(pddx_arglist, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    with subprocess.Popen(pddx_arglist, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as cav:
        print(cav)
        for row in gfg.gfg_list:
            (x, y, z, sigma, epsilon) = row
#        print(f"x={x} y={y} z={z} sigma={sigma} epsilon={epsilon}", file=cav.stdin)
            tab = str("\t")
            newline = str("\n")
            line = str(x)+tab+str(y)+tab+str(z)+tab+str(sigma)+tab+str(epsilon)+newline
            print("FTW: " + line)
#            print(bytes((str(x)+tab+str(y)+tab+str(z)+tab+str(sigma)+tab+str(epsilon))+newline, encoding='utf-8'))
#            cav.stdin.write(bytes((str(x)+tab+str(y)+tab+str(z)+tab+str(sigma)+tab+str(epsilon))+newline, encoding='utf-8'))
            cav.stdin.write(bytes(line, encoding='utf-8'))

        cav.stdin.close()

        # capture output
#        for line in cav.stdout:
#            print(line.rstrip().decode('ASCII'))
#            print(line)

        # Create the cav object from stdout
        print("building retval:")
        for line in cav.stdout:
            dsline = line.decode()
            print("FTW dsline: " + dsline)
#            print("FTW line: " + str(line))
#rstrip() removes trailing space (e.g. newline)
#            sline = str(line)
#            rline = line.rstrip()
#            dline = rline.decode('ASCII')
#            print("FTW dline: " + dline)
#            pddx_retval.append(dline)
            pddx_retval.append(dsline)
#            (x, y, z, d) = line.rstrip().decode('ASCII')
#            (x, y, z, d) = dline.split("\t")
#            print(f"{x}\t{y}\t{z}\t{d}")
        print("FTW sending retval:")
        print(pddx_retval[0])

        cav.stdout.close()

#        with open(input_file, "r", encoding="utf8") as gfg_file:
#            reader = csv.reader(gfg_file, delimiter="\t")
#            for row in reader:
#                (x, y, z, sigma, epsilon) = row
#                self.gfg_list.append(row);

    return pddx_retval

    # Build an object to return using the stream output
#    return cav(input_file=cav.stdout, dims=gfg.dims)
#class cav(VACUUMMS):
#    def __init__(self, input_file, dims):
