# This is vacuumms.py, defines the package 

import csv
import subprocess


## VACUUMMS types 


# superclass for all data objects

class VACUUMMS:

    def __init__(self, box):
        print("constructing VACUUMMS object")
        self.box = box

    def dump(self):
        print("# box: " + str(self.box)) 

    # Placeholder in case we factor parts out 
    def to_file(self, filename):
        pass

    # write contents as dataset to hdf5 formatted file
    def to_hdf5(self, hdf5_filename, hdf5_path, hdf5_dataset_name):
        pass


class cav_record():
    def __init__(self, record):
        (self.x, self.y, self.z, self.d) = record
    def __str__(self):
        tab = "\t"
        return str(self.x) + tab + str(self.y) + tab + str(self.z) + tab + str(self.d)

class cav(VACUUMMS):

    def __init__(self, input_source, box):
        super().__init__(box)
        print("constructing VACUUMMS::cav object")
        self.cav_list = []

        # Capture the output stream instead of a file to create the cav object

        if (str(type(input_source)) == "<class '_io.BufferedReader'>"):
            for line in input_source:
                self.cav_list.append(cav_record(line.decode().rstrip().split("\t")))

        else: # otherwise treat input_source as a file
            with open(input_file, "r", encoding="utf8") as cav_file:
                reader = csv.reader(cav_file, delimiter="\t")
                for row in reader:
                    self.cav_list.append(cav_record(row))

    def dump(self):
        super().dump() # Dump PVT, box, etc
        for record in self.cav_list:
            print(f"{record.x}\t{record.y}\t{record.z}\t{record.d}")

    # write contents to flat file (traditional)
    def to_file(self, filename="/dev/null"):
        with open(filename, "w") as file:
            for record in self.cav_list:
                print(f"{record.x}\t{record.y}\t{record.z}\t{record.d}", file=file)


class gfg_record():
    def __init__(self, record):
        (self.x, self.y, self.z, self.sigma, self.epsilon) = record

class gfg(VACUUMMS):
    def __init__(self, input_file, box):
        super().__init__(box)
        print("constructing VACUUMMS::gfg object from "+input_file + " with box = " + str(box))
        self.gfg_list = []

        with open(input_file, "r", encoding="utf8") as gfg_file:
            reader = csv.reader(gfg_file, delimiter="\t")
            for row in reader:
                self.gfg_list.append(gfg_record(row))

    def dump(self):
        super().dump() # Dump PVT, box, etc
        for record in self.gfg_list:
            print(f"{record.x}\t{record.y}\t{record.z}\t{record.sigma}\t{record.epsilon}")


## VACUUMMS methods


def ddx(gfg, box=[6.0, 6.0, 6.0],
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

