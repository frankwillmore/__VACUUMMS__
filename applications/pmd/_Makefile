#!/usr/bin/make

HOME=/home1/00479/willmore

bmd:	bmd.c 
	gcc -g bmd.c -o bmd -lftw -lm -I$(HOME)/usr/include -L$(HOME)/usr/lib 
	mv bmd $(HOME)/usr/bin/

pmd:	pmd.c 
	gcc -g pmd.c -o pmd -lftw -lm -I$(HOME)/usr/include -L$(HOME)/usr/lib 
	mv pmd $(HOME)/usr/bin/

sdlj:	sdlj.c 
	gcc sdlj.c -o sdlj -lftw -lm -I/home/willmore/usr/include -L/home/willmore/lib 
	mv sdlj /home/willmore/bin/sdlj

allgo:	allgo.c 
	gcc allgo.c -o allgo -lftw -lm -I/home/willmore/usr/include -L/home/willmore/lib 
	mv allgo /home/willmore/bin/allgo

sdlj-mfp:	sdlj-mfp.c 
	gcc sdlj-mfp.c -o sdlj-mfp -lftw -lm -I/home/willmore/usr/include -L/home/willmore/lib 
	mv sdlj-mfp /home/willmore/bin/sdlj-mfp

sdlj2p:	sdlj2p.c 
	gcc sdlj2p.c -o sdlj2p -lftw -lm -I/home/willmore/usr/include -L/home/willmore/lib 
	mv sdlj2p /home/willmore/bin/sdlj2p


