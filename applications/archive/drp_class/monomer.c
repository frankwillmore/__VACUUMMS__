/* this is the first of 3 programs for determining the change in excess */
/* Gibbs free energy for mixing of hard sphere monomer with hard sphere */
/* chains by means of calculating insertion probabilities.  */
/* This program in particular calculates G/kT for the monomer */
/* */ 

#include "../general/graphics.c"
#include "../general/rng.c"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define N 1000 /*max # of objects*/
#define PI 3.14159
#define WSIZE 512

double R=.02;  /*circle radius*/
double D2; /* diameter squared */
double eta = 0.15; /* occupied volume/total volume */
int n; /*number of objects*/
double X[N], Y[N];

int main(int argc, char *argv[])
{
  int i;
  int attempts=0;
  int successes=0;

  n = floor(eta / (PI * R * R));
  if (n>N) 
  {
    printf("too many particles: n=%d\n", n);
    exit(0);
  }

  printf("using %d particles for actual eta = %lf\n", n, (double)(n*PI*R*R));

  startgraphics();
  randinit();

  D2=4*R*R;

  /* insert n particles */
  for (i=0; i<n;)
  {
    double testx, testy;
    testx=RND();
    testy=RND();
    if (Pinsert(testx, testy))
    {
      X[i]=testx;
      Y[i]=testy;
 //     printf("%d:  inserted at %lf, %lf\n",i, testx,testy);
      i++;
    }
  }
//printf("i particles: %d\n", i); 
  for (attempts=0; attempts<1000; attempts++)
  { 
    double x=RND();
    double y=RND();
    successes+=Pinsert(x,y);
    drawObjects();
    check4event();
//printf("attempts: %d\n", attempts);
  }

  printf("value of B = %lf\n", (double)((0.0+successes)/attempts));
} /* end main */

int Pinsert(double x, double y)
{
  int i;
  for (i=0; i<n; i++)
  {
    if (checkOverlap(i, x, y)) return 0;
  }
  return 1;
}

int checkOverlap(int n1, double x, double y)
{ 
  // returns 0 if x,y doesn't overlap n1
  double r2, dx, dy;

  box5:
  {
    dx = X[n1] - x;
    dy = Y[n1] - y;
    r2=dx*dx + dy*dy;
    if (r2 < D2) return 1;
  }
  

  box1:
  {
    dx = X[n1] - 1 - x;
    dy = Y[n1] - 1 - y;
    r2=dx*dx + dy*dy;
    if (r2 < D2) return 1;
  }

  box2:
  {
    dx = X[n1] - x;
    dy = Y[n1] - 1 - y;
    r2=dx*dx + dy*dy;
    if (r2 < D2) return 1;
  }

  box3:
  {
    dx = X[n1] + 1 - x;
    dy = Y[n1] - 1 - y;
    r2=dx*dx + dy*dy;
    if (r2 < D2) return 1;
  }
  box4:
  {
    dx = X[n1] - 1 - x;
    dy = Y[n1] - y;
    r2=dx*dx + dy*dy;
    if (r2 < D2) return 1;
  }

  box6:
  {
    dx = X[n1] + 1 - x;
    dy = Y[n1] - y;
    r2=dx*dx + dy*dy;
    if (r2 < D2) return 1;
  }

  box7:
  {
    dx = X[n1] - 1 - x;
    dy = Y[n1] + 1 - y;
    r2=dx*dx + dy*dy;
    if (r2 < D2) return 1;
  }

  box8:
  {
    dx = X[n1] - x;
    dy = Y[n1] + 1 - y;
    r2=dx*dx + dy*dy;
    if (r2 < D2) return 1;
  }

  box9:
  {
    dx = X[n1] + 1 - x;
    dy = Y[n1] + 1 - y;
    r2=dx*dx + dy*dy;
    if (r2 < D2) return 1;
  }

  return 0;
}

void drawObjects()
{
  int i=0;
  gcvalues.foreground = BG_COLOR;
  XChangeGC(dpy, context, GCForeground, &gcvalues);
  XFillRectangle(dpy, window, context, 0, 0, WSIZE*2, WSIZE*2);
  
  gcvalues.foreground = FG_COLOR;
  XChangeGC(dpy, context, GCForeground, &gcvalues);
  for(i=0;i<n;i++)
  {
    int XX, YY;
    XX=X[i]*WSIZE;
    YY=Y[i]*WSIZE;
    XFillArc(dpy, window, context, XX-R, YY-R, R*2*WSIZE, R*2*WSIZE, 0, 360*64);
  }
  XFlush(dpy);
}

