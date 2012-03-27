#!/usr/bin/make

HOME_DIR=/home/willmore

end2end:	end2end.c
	gcc end2end.c -I/home/willmore/usr/include -L/home/willmore/lib -lm -lftw -o end2end

install:	end2end
	mv end2end $(HOME_DIR)/bin
