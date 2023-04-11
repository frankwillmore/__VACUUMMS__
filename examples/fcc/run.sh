#!/bin/sh

# The example configuration contains 108 atoms. There are 4 atoms in a unit cell, so 108/4 = 27
# 3^d = 27, so this is unit cell repeated three deep in each direction
# The atoms here are unit diameter, so unit cell dimension for FCC is sqrt(2) and box dim is 3*sqrt(2) = 4.24264

gfg2fvi -box 4.24264 4.24264 4.24264   \
        -resolution 256 \
        -potential 612 \
        < fcc.gfg \
        > fcc.fvi
