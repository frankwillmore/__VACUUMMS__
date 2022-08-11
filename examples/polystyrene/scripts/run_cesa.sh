pddx \
     -seed 5678 \
     -n_samples 1000000 \
     -n_threads 16 \
     -box 87.57884525964772 87.57884525964772 87.57884525964772 \
     < PS.gfg \
     >> PS_1M.cav
#     -n_steps 1000000 \
#     -verlet_cutoff 100.0 \
#     -seed 1357 \
#     -precision_parameter 0.1 \
#     < PS_trunc.gfg
#     < PS_Corrected_crammed.gfg
#     -box 87.57884525964772 87.57884525964772 87.57884525964772 \
#     -box 120 120 120 \

#usage:	-box [ 6.0 6.0 6.0 ]
#		-seed [ 1 ]
#		-characteristic_length [ 1.0 ] if in doubt, use largest sigma/diameter
#		-characteristic_energy [ 1.0 ] if in doubt, use largest energy/epsilon
#		-show_steps (includes steps taken as final column)
#		-verlet_cutoff [ 100.0 ]
#		-n_samples [ 1 ]
#		-volume_sampling 
#		-include_center_energy 
#		-min_diameter [ 0.0 ] 0.0 will give all 

