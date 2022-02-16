#!/usr/bin/make

HOME=/share/home/00479/willmore/

utrw:	utrw.c 
	gcc utrw.c -o utrw -lftw -lm -I/home/willmore/usr/include -L/home/willmore/lib 
	cp utrw /home/willmore/bin/utrw

utrwt:	utrwt.c 
	gcc utrwt.c -o utrwt -lftw -lm -I/home/willmore/usr/include -L/home/willmore/lib 
	cp utrwt /home/willmore/bin/utrwt

#utrwt69:	utrwt69.c 
#	gcc utrwt69.c -o utrwt69 -lftw -lm -I/home/willmore/usr/include -L/home/willmore/lib 
#	cp utrwt69 /home/willmore/bin/utrwt69

utrwt69:	utrwt69.c 
	icc -g -qp utrwt69.c -o utrwt69 -lftw -lm -I$(HOME)/usr/include -L$(HOME)/lib 

utrwt69-debug:	utrwt69.c 
	gcc -g utrwt69.c -o utrwt69-debug -lftw -lm -I$(HOME)/usr/include -L$(HOME)/lib 

utrwt69u:	utrwt69u.c 
	gcc utrwt69u.c -o utrwt69u -lftw -lm -I/home/willmore/usr/include -L/home/willmore/lib 
	cp utrwt69U /home/willmore/bin/utrwt69u

