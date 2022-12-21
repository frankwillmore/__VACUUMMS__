# This is vacuumms.py, defines the package 

import csv
import subprocess

#def v_drain_pipe(pipe="", file="/dev/null"):
#    # Drain a pipe to a file ?
#    f=open("file", "w")
#    for line in pipe.stdout:
#        f.write(line)
#        print(line.rstrip().decode('ASCII'))

# open the file in dst2hst. Can fix stderr later. 
#def dst2hst():
#    f=open('x.dst', "r")
#    d2h=subprocess.Popen("dst2hst", stdout=subprocess.PIPE, stdin=f)
# Just dump pipe to stdout
#    for line in d2h.stdout:
#        print(line.rstrip().decode('ASCII'))
    


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

    def __init__(self, input_file, dims):
        super().__init__(dims)
        print("constructing VACUUMMS::cav object")
        self.cav_list = []

        with open(input_file, "r", encoding="utf8") as cav_file:
            reader = csv.reader(cav_file, delimiter="\t")
            for row in reader:
                self.cav_list.append(cav_record(row))
#                (x, y, z, d) = row
#                tuple = [x, y, z, d]
#                self.cav_list.append(tuple)

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
#                (x, y, z, sigma, epsilon) = row
#                self.gfg_list.append(row);

    def dump(self):
        super().dump() # Dump PVT, dims, etc
        for record in self.gfg_list:
            print(f"{record.x}\t{record.y}\t{record.z}\t{record.sigma}\t{record.epsilon}")
#        for row in self.gfg_list:
#            (x, y, z, sigma, epsilon) = row
#            print(f"x={x} y={y} z={z} sigma={sigma} epsilon={epsilon}")


def ddx(gfg, n_steps=1000000, verlet_cutoff=100.0, n=1, min_diameter=0.0):
#    f=open('x.dst', "r")
#    d2h=subprocess.Popen("dst2hst", stdout=subprocess.PIPE, stdin=f)
#    wc = subprocess.Popen(['wc', '-l'], stdin=subprocess.PIPE)

    ddx_retval = []
#    ddx_arglist = ['ddx', '-box ', str(gfg.dims[0]), str(gfg.dims[1]), str(gfg.dims[2]) ]
    ddx_arglist = ['ddx', '-box ', str(gfg.dims[0]), str(gfg.dims[1]), str(gfg.dims[2]) ]
    ddx_arglist.append("-n_steps " + str(n_steps))
    print("FTW ddx_arglist: " + str(ddx_arglist))

    # open the process for ddx, write info, and capture output
    # cav = subprocess.Popen(ddx_arglist, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    with subprocess.Popen(ddx_arglist, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as cav:
        print(cav)
        for row in gfg.gfg_list:
#            (x, y, z, sigma, epsilon) = row
#        print(f"x={x} y={y} z={z} sigma={sigma} epsilon={epsilon}", file=cav.stdin)
            tab = str("\t")
            newline = str("\n")
            line = str(row.x)+tab+str(row.y)+tab+str(row.z)+tab+str(row.sigma)+tab+str(row.epsilon)+newline
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
#            ddx_retval.append(dline)
            ddx_retval.append(dsline)
#            (x, y, z, d) = line.rstrip().decode('ASCII')
#            (x, y, z, d) = dline.split("\t")
#            print(f"{x}\t{y}\t{z}\t{d}")
        print("FTW sending retval:")
        print(ddx_retval[0])

        cav.stdout.close()

#        with open(input_file, "r", encoding="utf8") as gfg_file:
#            reader = csv.reader(gfg_file, delimiter="\t")
#            for row in reader:
#                (x, y, z, sigma, epsilon) = row
#                self.gfg_list.append(row);

    return ddx_retval

    # Build an object to return using the stream output
#    return cav(input_file=cav.stdout, dims=gfg.dims)
#class cav(VACUUMMS):
#    def __init__(self, input_file, dims):
