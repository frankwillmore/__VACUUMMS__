#!/bin/make

# This is my file for exporting a set of codes (under target public) to the directory .public.  Git is instructed to ignore .public by naming it in .gitignore

public: 
	mkdir -p .public
	cp -R utils .public
	cp -R libraries .public
	cp -R applications/ddx .public/applications
	cp -R applications/rattle .public/rattle

clean:
	rm -rf .public

