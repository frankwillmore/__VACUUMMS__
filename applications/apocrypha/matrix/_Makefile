#!/usr/bin/make

HOME_DIR=/home1/00479/willmore

matrix:	matrix.c
	gcc matrix.c -I$(HOME_DIR)/usr/include -L$(HOME_DIR)/usr/lib -lftw -lm -o matrix
	mv matrix $(HOME_DIR)/usr/bin

xmatrix:	xmatrix.c
	gcc xmatrix.c -I$(HOME_DIR)/usr/include -L$(HOME_DIR)/usr/lib -lftw -lm -o xmatrix
	mv xmatrix $(HOME_DIR)/usr/bin

xmatrix2d:	xmatrix2d.c
	gcc xmatrix2d.c -I$(HOME_DIR)/usr/include -L$(HOME_DIR)/usr/lib -lftw -lm -o xmatrix2d
	mv xmatrix2d $(HOME_DIR)/usr/bin

matrix2d:	matrix2d.c
	gcc matrix2d.c -I$(HOME_DIR)/usr/include -L$(HOME_DIR)/usr/lib -lftw -lm -o matrix2d
	mv matrix2d $(HOME_DIR)/usr/bin
