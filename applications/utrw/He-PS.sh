#!/bin/bash

INDIR=/archive/utexas/ch/willmore/materials/polystyrene/gfg
OUTDIR=/work/willmore/polystyrene/He

box_diameter=25.5252

for i in 1 2 3 4 5 6 7 8 9 10
do
  nohup bsub -I -n 1 -q serial -M 500000 -W 3:30 -i $INDIR/$i.gfg -o $OUTDIR/$i.hst -e $HOME/$configuration_number.err pam -g 1 gmmpirun_wrapper /home/willmore/bin/utrwt69 < $INDIR/$i.gfg -box $box_diameter $box_diameter $box_diameter -randomize -n 100 -target_time 1000 -time_series -bin_size 25 -test_diameter 2.9 -test_epsilon 2.52 &

done
