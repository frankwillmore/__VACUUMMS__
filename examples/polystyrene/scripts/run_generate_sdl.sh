#!/bin/bash

box_dims=" -box 87.57884525964772 87.57884525964772 87.57884525964772 "

# Generate SDL for each atom type, 

# Hydrogen
grep 2.373410 PS.gfg \
    | gfg2pov \
    -color White \
    -clip \
    ${box_dims} \
    -camera 100.0 120.0 140.0 \
    -phong 0.7 \
    -shadowless \
    -background SkyBlue \
    > PS.pov 

# Little Carbon
grep 3.520530 PS.gfg \
    | gfg2pov \
    -color Black \
    -clip \
    ${box_dims} \
    -phong 1.0 \
    -shadowless \
    -no_header \
    >> PS.pov 

# Big Carbon
grep 3.581180 PS.gfg \
    | gfg2pov \
    -color Black \
    -clip \
    ${box_dims} \
    -phong 1.0 \
    -shadowless \
    -no_header \
    >> PS.pov 

cp PS.pov PS_plus_cav.pov

# now show the cavities
cav2pov \
    -transmit 0.5 \
    -phong 0.7 \
    -color Green \
    -clip \
    ${box_dims} \
    -no_header \
    < PS_1M.unq \
    >> PS_plus_cav.pov 

# finally, just the cavities
cav2pov \
    -transmit 0.5 \
    -color Green \
    -phong 0.7 \
    ${box_dims} \
    -clip \
    -camera 100.0 120.0 140.0 \
    -background SkyBlue \
    < PS_1M.unq \
    > PS_just_cav.pov 

# render the scenes

povray -W1920 -H1080 PS.pov
povray -W1920 -H1080 PS_plus_cav.pov
povray -W1920 -H1080 PS_just_cav.pov

# display the rendered output

feh PS.png &
feh PS_plus_cav.png &
feh PS_just_cav.png &
