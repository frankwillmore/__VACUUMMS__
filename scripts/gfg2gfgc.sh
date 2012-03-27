#!/bin/bash

#  This will read in a .gfg format and convert it to the atoms/diameters in PDMS

cat ~/ranger_home/materials/PDMS/gfg/PDMS_1.gfg | sed -e "s/0.023000/0.023000\tWhite\tH/" \
                | sed -e "s/0.198000/0.198000\tGray\tSi/" \
                | sed -e "s/0.062000/0.062000\tBlack\tC/" \
		| sed -e "s/0.080000/0.080000\tRed\tO/"   \
                | sed -e "s/0.008000/0.008000\tWhite\tH/" > 1.gfgci
