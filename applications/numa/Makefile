#!/bin/make

numa:		numa.cu 
		nvcc numa.cu -l cudart -lm -o numa 

numa-emu:		numa.cu 
		nvcc numa.cu -l cudart -lm -o numa-emu -deviceemu

