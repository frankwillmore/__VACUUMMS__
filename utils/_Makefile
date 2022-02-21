#!/bin/make

# Utilities install to home directory by default.  Change here for alternate configuration

HOME_DIR:=$$HOME

INCLUDE_DIR:=$(HOME_DIR)/usr/include
INSTALL_DIR:=$(HOME_DIR)/usr/bin
LIBRARY_DIR:=$(HOME_DIR)/usr/lib

# Change the following, depending on your CUDA installation...

CUDA_LIBRARY_DIR:=$(TACC_CUDA_LIB)
CUDA_INCLUDE_DIR:=$(TACC_CUDA_INC)
CUDA_BINARY_DIR:=$(TACC_CUDA_BIN)

# Change compiler info if you dare...

CC:=gcc
CFLAGS:=-L$(LIBRARY_DIR) -I$(INCLUDE_DIR) -lftw -lftw_pov -ltiff -lftw_tiff -lm 

NVCC:=$(CUDA_BINARY_DIR)/nvcc

# Probably won't want to change anything below here.

UTILS:=vis2pov cav2vis cfg2vis add dst2hst uniq avg a2b clustercat smooth normalize cfg2gfg nsplit fvi2tiff loge log10 exp sum pow sq ck cram wdst2hst max min miss std sew povheader gfg2pov gfgc2pov fvi2pov cav2pov stack_tiffs expr_add expr_multiply dwf sqrt stream2slice truncate stream_multiply replicate_gfg replicate_cav sgfg2fvi

%: %.c
	$(CC) $(CFLAGS) $< -o $@

CUDA_UTILS:=gfg2fvi ftw_cuda

gfg2fvi: gfg2fvi.c
	$(NVCC) gfg2fvi.c -L$(LIBRARY_DIR) -I$(INCLUDE_DIR) -L$(CUDA_LIBRARY_DIR) -I$(CUDA_INCLUDE_DIR) -lftw -lftw_cuda -lm -o gfg2fvi 

ftw_cuda:        ftw_cuda.c 
	$(NVCC) ftw_cuda.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -L$(CUDA_LIBRARY_DIR) -I$(CUDA_INCLUDE_DIR) -lftw -lftw_cuda -lm -o ftw_cuda 

.PHONY: utils cuda-utils install install-cuda everything clean

utils: $(UTILS)

cuda-utils: $(CUDA_UTILS)

install: utils
	mv $(UTILS) $(INSTALL_DIR)

install-cuda:	cuda-utils
	@echo installing cuda utils...
	mv $(CUDA_UTILS) $(INSTALL_DIR)

everything:  utils cuda-utils

clean:  
	rm -f $(UTILS) 
	rm -f $(CUDA_UTILS)

