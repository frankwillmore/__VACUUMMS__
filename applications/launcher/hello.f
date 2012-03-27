   Program HELLO
      Integer :: i, argc
      Character :: argv*100

      argc=iargc()
      if (argc==0) then
         print*, "Usage: hello <number>"
      else
         call getarg(1,argv)
         read(argv,*) id
         print*, "Hello, world! from ", id
      end if

   End Program HELLO
