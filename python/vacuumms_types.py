# This is vacuumms_types.py, defines the types 

import csv
import subprocess


## VACUUMMS types 


# superclass for all data objects

class VACUUMMS:

    def __init__(self, box):
        print("constructing VACUUMMS object")
        self.list = []
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

        # Capture the output stream instead of a file to create the cav object

        if (str(type(input_source)) == "<class '_io.BufferedReader'>"):
            for line in input_source:
                self.list.append(cav_record(line.decode().rstrip().split("\t")))

        else: # otherwise treat input_source as a file
            with open(input_file, "r", encoding="utf8") as cav_file:
                reader = csv.reader(cav_file, delimiter="\t")
                for row in reader:
                    self.list.append(cav_record(row))

    def dump(self):
        super().dump() # Dump PVT, box, etc
        for record in self.list:
            print(f"{record.x}\t{record.y}\t{record.z}\t{record.d}")

    # write contents to flat file (traditional)
#    def to_file(self, filename="/dev/null"):
    def to_file(self, filename=None):
        if (filename == None): filename=str(self) + ".cav"
        with open(filename, "w") as file:
            for record in self.list:
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


class cls_record():
    def __init__(self, record):
        (self.cluster, self.x, self.y, self.z, self.d) = record
    def __str__(self):
        tab = "\t"
        return str(self.cluster) + tab + str(self.x) + tab + str(self.y) + tab + str(self.z) + tab + str(self.d)

class cls(VACUUMMS):

    def __init__(self, input_source, box):
        super().__init__(box)
        print("constructing VACUUMMS::cls object")

        # Capture the output stream instead of a file to create the cls object

        if (str(type(input_source)) == "<class '_io.BufferedReader'>"):
            for line in input_source:
                self.list.append(cls_record(line.decode().rstrip().split("\t")))

        else: # otherwise treat input_source as a file
            with open(input_file, "r", encoding="utf8") as cls_file:
                reader = csv.reader(cls_file, delimiter="\t")
                for row in reader:
                    self.list.append(cls_record(row))

    def dump(self):
        super().dump() # Dump PVT, box, etc
        for record in self.list:
            print(f"{record.cluster}\t{record.x}\t{record.y}\t{record.z}\t{record.d}")

    # write contents to flat file (traditional)
    def to_file(self, filename=None):
        if (filename == None): filename=str(self) + ".cls"
        with open(filename, "w") as file:
            for record in self.list:
                print(f"{record.cluster}\t{record.x}\t{record.y}\t{record.z}\t{record.d}", file=file)


class fvi_record():
    def __init__(self, record):
        (self.x, self.y, self.z, self.energy) = record
    def __str__(self):
        tab = "\t"
        return str(self.x) + tab + str(self.y) + tab + str(self.z) + tab + str(self.energy)

class fvi(VACUUMMS):

    def __init__(self, input_source, box):
        super().__init__(box)
        print("constructing VACUUMMS::fvi object")

        # Capture the output stream instead of a file to create the fvi object

        if (str(type(input_source)) == "<class '_io.BufferedReader'>"):
            for line in input_source:
                self.list.append(fvi_record(line.decode().rstrip().split("\t")))

        else: # otherwise treat input_source as a file
            with open(input_file, "r", encoding="utf8") as fvi_file:
                reader = csv.reader(fvi_file, delimiter="\t")
                for row in reader:
                    self.list.append(fvi_record(row))

    def dump(self):
        super().dump() # Dump PVT, box, etc
        for record in self.list:
            print(f"{record.x}\t{record.y}\t{record.z}\t{record.energy}")

    # write contents to flat file (traditional)
    def to_file(self, filename=None):
        if (filename == None): filename=str(self) + ".fvi"
        with open(filename, "w") as file:
            for record in self.list:
                print(f"{record.x}\t{record.y}\t{record.z}\t{record.energy}", file=file)


