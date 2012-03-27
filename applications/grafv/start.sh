 bsub -I -W0:01 -q development -n 4 pam -g 1 gmmpirun_wrapper greetings
mpicc -o greetings greetings.c -lmpi
mpicc -o greetings greetings.c -L/home/willmore/lib -lftw -I/home/willmore/usr/include
