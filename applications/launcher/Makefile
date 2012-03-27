#----------------------------
# Standard for AIX (longhorn)
#----------------------------
#FC       = xlf90
#FC_FLAGS = 
#CC       = mpcc
#-----------------------------

#----------------------------
# Standard for IA32 (lonestar)
#----------------------------
FC       = ifc
FFLAGS = -FR -Vaxlib
CC       = mpicc
CFLAGS   =
#-----------------------------


all: mxm hello launcher

hello: hello.o
	$(FC) $(FFLAGS) -o $@ $<

mxm: mxm.o
	$(CC) $(CFLAGS) -o $@ $<

launcher: launcher.o

clean: 
	rm -f *.o mxm hello launcher
