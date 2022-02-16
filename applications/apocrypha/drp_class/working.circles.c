#include "../general/graphics.c"
#include "../general/rng.c"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#define N 256 /*max # of objects*/

double kB=1.381e-23;
double T=298.0;
double L=500;  /*box dimension*/
double R=20;  /*circle radius*/
double D2; /* diameter squared */
double plength=2; /* peturbation length */

int n=0; /*number of objects*/

double X[N], Y[N];

int counter=0;

int main(int argc, char *argv[])
{
  double x,y;
  int i;
  int xyz=0;

  startgraphics();
  randinit();

  D2=4*R*R;

  for (i=0; n<N; i++)
  {
    tryInsert();
    perturb();
    if (counter>5500)
    {
      drawObjects();
      check4event();
      counter=0;
  printf(".");
    }
  }

  while(1)
  {
    drawObjects();
    check4event();
    sleep(1);
  }
}

int tryInsert()
{
  double testx, testy;
  testx=RND()*L;
  testy=RND()*L;
  if (Pinsert(testx, testy))
  {
    X[n]=testx;
    Y[n]=testy;
    printf("%d:  inserted at %lf, %lf\n",n, testx,testy);
    n++;
    return 1;
  }
  else return 0;
}

void perturb()
{ 
  int i;
  int m;
  double deltax, deltay;
  double newx, newy;

  counter++;

  m = RND()*n;
  deltax =RND()*plength-plength/2;
  deltay =RND()*plength-plength/2;
  newx=X[m]+deltax;
  newy=Y[m]+deltay;

  if (newx>L) return;
  if (newy>L) return;
  if (newx<0) return;
  if (newy<0) return;

  for (i=0; i<n; i++)
  {
    if (i!=m)
    if (checkOverlap(i, newx, newy))
    {
      //printf("perturbation failed\n");
      return;
    }
  }

  // success:
  X[m] = newx;
  Y[m] = newy;
  //printf("!");
  //printf("%d:  success at %lf, %lf\n", n, newx, newy);
  return;
}

int Pinsert(double x, double y)
{
  int i,j;
  for (i=0; i<=n; i++)
  {
    if (checkOverlap(i, x, y)) return 0;
  }
  return 1;
}

int checkOverlap(int n1, double x, double y)
{ 
  // returns 0 if x,y doesn't overlap n1
  double r2, dx, dy;
    dx = X[n1] - x;
    dy = Y[n1] - y;
    r2=dx*dx + dy*dy;
    if (r2 < D2) 
    {
      return 1;
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
    XX=X[i];
    YY=Y[i];
    //  printf("drawing at %d, %d\n", XX, YY);
    XFillArc(dpy, window, context, XX-R, YY-R, R*2, R*2, 0, 360*64);
  }
  XFlush(dpy);
}

