#!/bin/make

# This is my file for exporting a set of codes (under target public) to the directory .public.  Git is instructed to ignore .public by naming it in .gitignore

public: 
	mkdir -p .public

fvt:
	mkdir -p .public/fvt
	cp -R utils .public/fvt
	cp -R libraries .public/fvt
	cp -R applications/ddx .public/fvt/applications
	cp -R applications/rattle .public/fvt/rattle

clean:
	rm -rf .public

