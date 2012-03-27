#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#define n 1200
double A[n][n], B[n][n], C[n][n];

int main(int argc, char* argv[])
{
   double sum, correct, alpha=1.0, beta=0.0, err=0.0, thresh=1.0e-09;
   int i, j, k;
   int idiag, ierr, npes, irank, MPI_COMM_WORLD;


   irank=atoi(argv[1]);
   for (i=0; i<n; i++)
   {
      for (j=0; j<n; j++)
      {
	 A[i][j]=10.0*i+j;
	 B[i][j]=0.0;
	 C[i][j]=0.0;
      };
      B[i][i]=1.0;
   };
   for (i=0; i<n; i++)
   {
      for (j=0; j<n; j++)
      {
         sum=0.0;
         for (k=0; k<n; k++)
	 {
	    sum+=A[i][k]*B[k][j];
	 };
         C[i][j]=sum;
      };
   };

   for (i=0; i<n; i++)
   {
      for (j=0; j<n; j++)
      {
	 correct=10.0*i+j;
	 err+=abs(C[i][j]-correct);
      }
   };
   if (err < thresh)
   {
      printf("Process %d: The program ran correctly\n",irank);
   }
   else
   {
      printf("Process %d: The program did not produce the correct results\n",irank);
   };
}
