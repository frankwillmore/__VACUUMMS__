#!/bin/bash

INDIR=/share/home/00479/willmore/materials/polystyrene/gfg

echo $INDIR

box_diameter=25.5252

./utrwt69-debug < /share/home/00479/willmore/materials/polystyrene/gfg/1.gfg \ 
-box $box_diameter $box_diameter $box_diameter \
-randomize \
-n 100 \
-target_time 1000 \
-time_series \
-bin_size 25 \
-test_diameter 2.9 \
-test_epsilon 2.52 &

