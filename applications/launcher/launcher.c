/*
 * Copyright (c) 2003 The Regents of the University of Texas System.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms are permitted provided
 * that the above copyright notice and this paragraph are duplicated in all
 * such forms and that any documentation, advertising materials,  and other
 * materials  related to such  distribution  and use  acknowledge  that the
 * software  was  developed  by the  University of Texas.  The  name of the
 * University may not be  used to endorse or promote  products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES OF
 * MERCHANTIBILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 */

/*-----------------------------------------------------------------------
 * Simple MPI Batch launcher for submitting multiple serial applications
 * for parametric studies. 
 *
 * Chona Guiang, TACC (Texas Advanced Computing Center) 
 * September, 2003
 *----------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define MAXSTR 100
#define MASTER 0

/*------------------------------------------------------------------------
 * Function declarations
 *------------------------------------------------------------------------*/
int my_system (char *command, char* envp[]);


int main(int argc, char* argv[], char* envp[])
{
   MPI_Comm COMM;
   int i, ierr, mype, npes, nparams, nstr;
   FILE *fptr;
   char **arglist;

   MPI_Init(&argc, &argv);
   COMM=MPI_COMM_WORLD;
   MPI_Comm_size(COMM,&npes);
   MPI_Comm_rank(COMM,&mype);

   if(mype == MASTER)
     if(npes <= 1)
       {
	 printf("Error: Unable to launch jobs (only 1 process available)\n");
	 return(1);
       }

   /* Allow for sequences of serial jobs (when number of jobs to
    * run is greater than the number of MPI procs) */
   
   nparams=npes;
   if (argc == 3) 
     nparams=atoi(argv[2]);

   /*
    * Allocate storage for character array that will hold serial commands
    */

   arglist    =(char **)malloc(nparams*sizeof(char *));
   arglist[0] =(char *)malloc(nparams*MAXSTR);

   for (i=1;i<nparams;i++) {
     arglist[i]=arglist[0]+i*MAXSTR;
   }

   if (mype==MASTER) {
     char *filename=(char *)malloc(MAXSTR);
     strcpy(filename,argv[1]);
     fptr=fopen(filename,"r");
     if (fptr==NULL) {
       printf("Cannot open file %s\n");
       return(1);
     }
     
     for (i=0;i<nparams;i++) {
       (void)fgets(arglist[i],MAXSTR,fptr);
     }
     
     free(filename);
     fclose(fptr);
   }
   
   nstr=nparams*MAXSTR;
   MPI_Bcast(arglist[0],nstr,MPI_CHAR,MASTER,COMM);
   
   /* Submit the jobs */
   
   for (i=mype;i<nparams;i+=npes) {
     my_system(arglist[i],envp);
   }
   
   /* Clean up */
   
   free(arglist[0]);
   free(arglist);
   
   MPI_Finalize();
   return(0);
}


int my_system (char *command, char* envp[]) {
  int pid, status, errno;
  
  if (command == 0)
    return 1;
  pid = fork();
  if (pid == -1)
    return -1;
  if (pid == 0) {

    /*
      Child process
    */

    char *argv[4];
    argv[0] = "sh";
    argv[1] = "-c";
    argv[2] = command;
    argv[3] = 0;
    execve("/bin/sh", argv, envp);
    exit(127);
  } 
  do {
    waitpid(pid, &status, 0);
    //		   printf("pid=%d    status=%d\n",pid,status);
    return status;
  } while(1);
}
