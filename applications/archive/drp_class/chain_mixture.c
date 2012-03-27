/* this is the second of 3 programs for determining the change in excess */
/* Gibbs free energy for mixing of hard sphere monomer with hard sphere */
/* chains by means of calculating insertion probabilities.  */
/* This program in particular calculates G/kT for the polymer chain. */
/* chains.c */ 

#include "../general/graphics.c"
#include "../general/rng.c"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define NMAX 1000 /* max # of monomers */
#define PI 3.14159
#define WSIZE 512 /* size of display window only */
#define DP1 9 /* degree of polymerization/size of chains */
#define DP2 1 /* degree of polymerization/size of chains */
#define R .015 /* monomer radius */
#define MAX_ATTEMPTS 100000 /* # of insertion attempts */

double D2=R*R*4; /* diameter squared */
double phi1=.5, phi2=.5; /* volume fraction */
double eta = 0.15; /* occupied volume/total volume */
int Nm=0; /* number of monomer units in system */
int Nc1, Nc2; /* number of chains */
double X[NMAX], Y[NMAX];
int graphics = 1;  /* run display or not */

int main(int argc, char *argv[])
{
  int i1, i2, j, k; /* loop indices */
  int successes1=0, successes2=0;
  int attempts;

  Nc1 = floor(phi1*eta / (PI * R * R * DP1));
  Nc2 = floor(phi2*eta / (PI * R * R * DP2));
  printf("chains: Nc1=%d, Nc2=%d\n", Nc1, Nc2);
  //printf("using %d chains for actual eta = %lf\n", Nc, (double)(DP*Nc*PI*R*R));

  if (graphics) startgraphics();
  randinit();

  /* insert Nc1 chains */
  i1 = 0;
  loop_insert1: while(i1<Nc1) // for each chain
  {
    double end_x, end_y;

    // first drop chain end
    end_x=RND();
    end_y=RND();
    if (Pinsert(end_x, end_y))
    {
      X[Nm]=end_x;
      Y[Nm]=end_y;
      Nm++;
    }
    else continue;

    // then grow the chain....    
    loop_chain:for (j=1; j<DP1;) // for each monomer in chain
    {
      double test_theta = RND()*PI*2;
      double deltaX, deltaY;
      double testx, testy;
      deltaX = 2*R * cos(test_theta);
      deltaY = 2*R * sin(test_theta);
      testx = end_x + deltaX;
      testy = end_y + deltaY; 
      if (Pinsert(testx, testy))
      {
        X[Nm] = testx;
        Y[Nm] = testy;
        end_x = testx;
        end_y = testy;
        Nm++;
        j++;
      }
      else
      {
         /* rollback */
         Nm-=j;
         goto loop_insert1; // insert failed, start over
      }
    } /* end loop_chain */
 
    drawObjects();
    check4event();

    i1++;

  } /* end loop_insert1 */

  /* insert Nc2 chains */
  i2 = 0;
  loop_insert2: while(i2<Nc2) // for each chain
  {
    double end_x, end_y;

    // first drop chain end
    end_x=RND();
    end_y=RND();
    if (Pinsert(end_x, end_y))
    {
      X[Nm]=end_x;
      Y[Nm]=end_y;
      Nm++;
    }
    else continue;

    // then grow the chain....    
    loop_chain2:for (j=1; j<DP2;) // for each monomer in chain
    {
      double test_theta = RND()*PI*2;
      double deltaX, deltaY;
      double testx, testy;
      deltaX = 2*R * cos(test_theta);
      deltaY = 2*R * sin(test_theta);
      testx = end_x + deltaX;
      testy = end_y + deltaY; 
      if (Pinsert(testx, testy))
      {
        X[Nm] = testx;
        Y[Nm] = testy;
        end_x = testx;
        end_y = testy;
        Nm++;
        j++;
      }
      else
      {
         /* rollback */
         Nm-=j;
         goto loop_insert2; // insert failed, start over
      }
    } /* end loop_chain2 */
 
    drawObjects();
    check4event();

    i2++;

  } /* end loop_insert2 */

printf("Nm = \%d\n", Nm);
/**************************************************************/
  printf("Calculating insertion probability 1...\n");
  fflush(stdout);

  insertion1:for (attempts=0; attempts<MAX_ATTEMPTS; attempts++)
  {
    double end_x, end_y;
 
    // first drop chain end
    end_x=RND();
    end_y=RND();
    if (Pinsert(end_x, end_y))
    {
      X[Nm]=end_x;
      Y[Nm]=end_y;
      Nm++;
    }
    else continue; // next attempt

    // then grow the chain....    
    for (j=1; j<DP1;) // for each monomer in chain
    {
      double test_theta = RND()*PI*2;
      double deltaX, deltaY;
      double testx, testy;
      deltaX = 2*R * cos(test_theta);
      deltaY = 2*R * sin(test_theta);
      testx = end_x + deltaX;
      testy = end_y + deltaY; 
      if (Pinsert(testx, testy))
      {
        X[Nm] = testx;
        Y[Nm] = testy;
        end_x = testx;
        end_y = testy;
        Nm++;
        j++;
      }
      else
      {
         /* restore Nm */
         Nm-=j;
         attempts++;
         goto end_insertion1; // insert failed, start over
      }
    } /* end loop_chain2 */

    successes1++;
    Nm-=DP1;
  
    end_insertion1:

  } /* end insertion1 */

printf("Nm = \%d\n", Nm);
printf("attempts: \%d\tsuccesses: %d\n", attempts, successes1);
fflush(stdout);

/**************************************************************/
  printf("Calculating insertion probability 2...\n");
  fflush(stdout);

  insertion2:for (attempts=0; attempts<MAX_ATTEMPTS; attempts++)
  {
    double end_x, end_y;
 
    // first drop chain end
    end_x=RND();
    end_y=RND();
    if (Pinsert(end_x, end_y))
    {
      X[Nm]=end_x;
      Y[Nm]=end_y;
      Nm++;
    }
    else continue; // next attempt

    // then grow the chain....    
    for (j=1; j<DP2;) // for each monomer in chain
    {
      double test_theta = RND()*PI*2;
      double deltaX, deltaY;
      double testx, testy;
      deltaX = 2*R * cos(test_theta);
      deltaY = 2*R * sin(test_theta);
      testx = end_x + deltaX;
      testy = end_y + deltaY; 
      if (Pinsert(testx, testy))
      {
        X[Nm] = testx;
        Y[Nm] = testy;
        end_x = testx;
        end_y = testy;
        Nm++;
        j++;
      }
      else
      {
         /* restore Nm */
         Nm-=j;
         attempts++;
         goto end_insertion2; // insert failed, start over
      }
    } /* end loop_chain2 */

    successes2++;
    Nm-=DP2;
  
    end_insertion2:
 
  } /* end insertion2 */

printf("Nm = \%d\n", Nm);
printf("attempts: \%d\tsuccesses: %d\n", attempts, successes2);
fflush(stdout);

} /* end main */

int Pinsert(double x, double y)
{
  int i;
  for (i=0; i<Nm; i++)
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
  for(i=0;i<Nm;i++)
  {
    int XX, YY;
    XX=X[i]*WSIZE;
    YY=Y[i]*WSIZE;
    XFillArc(dpy, window, context, XX-R, YY-R, R*2*WSIZE, R*2*WSIZE, 0, 360*64);
  }
  XFlush(dpy);
}

