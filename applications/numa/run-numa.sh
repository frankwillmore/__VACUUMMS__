#!/bin/bash

## Execute this file using qsub command:  qsub run-HAB6FDA

#$ -V                             # Inherit the submission environment
#$ -cwd                           # Start job in submission dir
#$ -N numa                     # Job name
#$ -j y                           # Combine stderr and stdout into stdout
#$ -o $JOB_NAME-$JOB_ID.hst       # Name of the output file
#$ -e $JOB_NAME-$JOB_ID.err       # Name of the stderr file
#$ -pe 1way 8                    # 8 cores/node on Longhorn
#$ -q normal                      # Queue name
#$ -P gpgpu                       # project type
#$ -A PMD-GPU                     # Account
#$ -l h_rt=0:05:00                # runtime (hh:mm:ss) - 6 hours max

numa 
