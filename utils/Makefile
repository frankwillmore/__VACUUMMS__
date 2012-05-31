#!/usr/bin/make

# I keep my libs on a branch off of $HOME

HOME:=/home1/00479/willmore/

INCLUDE_DIR:=$(HOME)/usr/include
INSTALL_DIR:=$(HOME)/usr/bin
LIBRARY_DIR:=$(HOME)/usr/lib

CC:=gcc
CFLAGS=-L$(LIBRARY_DIR) -I$(INCLUDE_DIR) -lftw -lftw_pov -ltiff -lftw_tiff -lm 

UTILS:=vis2pov cav2vis cfg2vis add dst2hst uniq avg a2b clustercat smooth normalize cfg2gfg nsplit fvi2tiff loge log10 exp sum pow sq ck cram wdst2hst max min miss std sew povheader gfg2pov gfgc2pov fvi2pov cav2pov stack_tiffs expr_add expr_multiply dwf sqrt stream2slice truncate stream_multiply

# need to fix:  replicate 

CUDA_UTILS:=replicate_gfg gfg2fvi ftw_cuda

#stack_tiffs:	stack_tiffs.c
#	$(CC) stack_tiffs.c -L$(LIBRARY_DIR) -I$(INCLUDE_DIR) -lftw -ltiff -lftw_tiff -lm -o stack_tiffs 

#readtiff:	readtiff.c
#	$(CC) readtiff.c -L$(LIBRARY_DIR) -I$(INCLUDE_DIR) -lftw -ltiff -lftw_tiff -lm -o readtiff 

#sum:	sum.c
#	$(CC) sum.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o sum

#pow:	pow.c
#	$(CC) pow.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o pow

#sq:	sq.c
#	$(CC) sq.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o sq

#ck:	ck.c
#	$(CC) ck.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o ck

#cram:	cram.c
#	$(CC) cram.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o cram

#cfg2gfg:	cfg2gfg.c
#	$(CC) cfg2gfg.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o cfg2gfg

#vis2pov:	vis2pov.c
#	$(CC) vis2pov.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o vis2pov

#cav2vis:	cav2vis.c
#	$(CC) cav2vis.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o cav2vis

#cfg2vis:	cfg2vis.c
#	$(CC) cfg2vis.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o cfg2vis

#gfg2pov:	gfg2pov.c
#	$(CC) gfg2pov.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -Wall -lftw_pov -lftw -lm -o gfg2pov

#add:	add.c
#	$(CC) add.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o add

#wdst2hst:        wdst2hst.c
#	$(CC) wdst2hst.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o wdst2hst 

#dst2hst:        dst2hst.c
#	$(CC) dst2hst.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o dst2hst 

#fvi2tiff:  fvi2tiff.c
#	$(CC) fvi2tiff.c -L$(LIBRARY_DIR) -I$(INCLUDE_DIR) -lftw -ltiff -lftw_tiff -lm -o fvi2tiff 

#uniq:        uniq.c
#	$(CC) uniq.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o uniq 

#max:	max.c
#	$(CC) max.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o max 

#min:	min.c
#	$(CC) min.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o min 

#miss:	miss.c
#	$(CC) miss.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o miss 

#std:	std.c
#	$(CC) std.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o std 

#nsplit:	nsplit.c
#	$(CC) nsplit.c -Wall -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o nsplit 

#avg:	avg.c
#	$(CC) avg.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o avg 

#loge:	loge.c
#	$(CC) loge.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o loge 

#log10:	log10.c
#	$(CC) log10.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o log10 

#exp:	exp.c
#	$(CC) exp.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o exp 

#replicate:        replicate.c
#	$(CC) replicate.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o replicate 

#a2b:	a2b.c
#	$(CC) a2b.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o a2b

#clustercat:	clustercat.c
#	$(CC) clustercat.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o clustercat

#smooth:	smooth.c
#	$(CC) smooth.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o smooth

#normalize:	normalize.c
#	$(CC) normalize.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o normalize

#sew:	sew.c
#	$(CC) sew.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o sew

#gfgc2pov: gfgc2pov.c
#	$(CC) gfgc2pov.c -Wall -lftw_pov -lm -lftw -L${LIBRARY_DIR} -I${INCLUDE_DIR} -o gfgc2pov

#povheader: povheader.c
#	$(CC) povheader.c -lm -lftw -L${LIBRARY_DIR} -I${INCLUDE_DIR} -o povheader

#cav2pov:	cav2pov.c
#	$(CC) cav2pov.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -lftw_pov -o cav2pov

#fvi2pov: fvi2pov.c
#	$(CC) fvi2pov.c -Wall -lftw_pov -lm -lftw -L${LIBRARY_DIR} -I${INCLUDE_DIR} -o fvi2pov

#expr_add:	expr_add.c
#	$(CC) expr_add.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o expr_add

#expr_multiply:	expr_multiply.c
#	$(CC) expr_multiply.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o expr_multiply

#dwf:	dwf.c
#	$(CC) dwf.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o dwf

#sqrt:	sqrt.c
#	$(CC) sqrt.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o sqrt 

#stream2slice:	stream2slice.c
#	$(CC) stream2slice.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o stream2slice 

#truncate:	truncate.c
#	$(CC) truncate.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o truncate 

#stream_multiply:	stream_multiply.c
#	$(CC) stream_multiply.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -o stream_multiply 

# CUDA utils

#test.o: test.cu
#	nvcc test.cu -c -I$(INCLUDE_DIR) 

#maltest:	maltest.cu
#	nvcc maltest.cu -L$(LIBRARY_DIR) -lm -lftw -I$(INCLUDE_DIR) -o maltest -lcudart

#test:	test.o test.c
#	nvcc test.c test.o -L$(LIBRARY_DIR) -lm -lftw -lftw_cuda -I$(INCLUDE_DIR) -o test -lcudart

#gfg2fvi: gfg2fvi.c
#	$(CC) gfg2fvi.c -lm -lftw -lftw_cuda -L$(LIBRARY_DIR) -I$(INCLUDE_DIR) -o gfg2fvi 

#replicate_gfg:        replicate_gfg.c
#	$(CC) replicate_gfg.c -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lm -lftw_cuda -o replicate_gfg 

#ftw_cuda:        ftw_cuda.c 
#	$(CC) ftw_cuda.c -o ftw_cuda -I$(INCLUDE_DIR) -L$(LIBRARY_DIR) -lftw -lftw_cuda -lm

%: %.cu

%: %.c
	$(CC) $(CFLAGS) $< -o $@

# dependencies and such
.PHONY: utils cuda-utils install install-cuda 

utils: $(UTILS)

cuda-utils: $(CUDA_UTILS)

install: utils
	mv $(UTILS) $(INSTALL_DIR)

install-cuda:	cuda-utils
	@echo installing cuda utils...
	mv $(CUDA_UTILS) $(INSTALL_DIR)

clean:  
	rm -f $(UTILS) 
	rm -f $(CUDA_UTILS)

