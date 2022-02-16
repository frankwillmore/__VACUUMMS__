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

#define RNG_SEED 24375 

#define NMAX 10000 /* max # of monomers */
#define PI 3.14159
#define WSIZE 512 /* size of display window only */
#define DP1 5 /* degree of polymerization/size of chains */
#define DP2 1 /* degree of polymerization/size of chains */
#define R .004 /* monomer radius */
#define MAX_ATTEMPTS 25000 /* # of insertion attempts */
#define NTRIALS 25 

double D2=R*R*4; /* diameter squared */
double phi1, phi2; /* volume fraction */
double eta = 0.15; /* occupied volume/total volume */
int Nm=0; /* number of monomer units in system */
//int Nc1, Nc2; /* number of chains */
double X[NMAX], Y[NMAX];
int graphics = 1;  /* run display or not */
double Gex[11];

int main(int argc, char *argv[])
{
  int i1, i2, j, k; /* loop indices */
  int successes1, successes2;
  int attempts;
  int phi_index;
  int trial_index;
  
  int B1[11], B2[11];
  int Nc1[11], Nc2[11];

printf("seed = %d\n", RNG_SEED);
  if (graphics) startgraphics(WSIZE);
  randinit(RNG_SEED);

/* loop over phi_index */

for (phi_index=0; phi_index<11; phi_index++)
{
printf("%d\n", phi_index);
fflush(stdout);
  B1[phi_index]=0;
  B2[phi_index]=0;
  Nc1[phi_index]=0;
  Nc2[phi_index]=0;
  Gex[phi_index]=0;

  phi1 = .1 * phi_index;
  phi2=1-phi1;

  Nm=0;
  Nc1[phi_index] = floor(phi1*eta / (PI * R * R * DP1));
  Nc2[phi_index] = floor(phi2*eta / (PI * R * R * DP2));

/*************************************************************************/
  /* insert Nc1 chains */
  i1 = 0;
  loop_insert1: while(i1<Nc1[phi_index]) // for each chain
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
      if (testx > 1) testx-=1;
      if (testx < 0) testx+=1;
      testy = end_y + deltaY; 
      if (testy > 1) testy-=1;
      if (testy < 0) testy+=1;
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
 
    if (graphics)
    {
      drawObjects();
      check4event();
    }

    i1++;

  } /* end loop_insert1 */

  /* insert Nc2 chains */
  i2 = 0;
  loop_insert2: while(i2<Nc2[phi_index]) // for each chain
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
      if (testx > 1) testx-=1;
      if (testx < 0) testx+=1;
      testy = end_y + deltaY; 
      if (testy > 1) testy-=1;
      if (testy < 0) testy+=1;
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
 
    if (graphics)
    {
      drawObjects();
      check4event();
    }

    i2++;

  } /* end loop_insert2 */

/*************** calculate B1 ***********************************************/

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
      if (testx > 1) testx-=1;
      if (testx < 0) testx+=1;
      testy = end_y + deltaY; 
      if (testy > 1) testy-=1;
      if (testy < 0) testy+=1;
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

    B1[phi_index]++;
    Nm-=DP1;
  
    end_insertion1:

  } /* end insertion1 */

/**************************************************************/

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
      if (testx > 1) testx-=1;
      if (testx < 0) testx+=1;
      testy = end_y + deltaY; 
      if (testy > 1) testy-=1;
      if (testy < 0) testy+=1;
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

    B2[phi_index]++;
    Nm-=DP2;

    end_insertion2:
  
  } /* end insertion2 */

} /* end looop over phi1 */

/* Results start here... */

printf("phi1\t\tNc1_tot\tB1\tNc2_tot\tB2\tGex\n");
for(phi_index=0; phi_index<11; phi_index++)
{
  int B1ref, B2ref;
  double G1, G2;
  double R1, R2;

  phi1 = .1 * phi_index;

  B1ref = B1[10];
  B2ref = B2[0];

  R1 = (0.0 + B1ref)/B1[phi_index];
  R2 = (0.0 + B2ref)/B2[phi_index];
  G1 = Nc1[phi_index]*log(R1);
  G2 = Nc2[phi_index]*log(R2);
  Gex[phi_index] = (G1 + G2)/(0.0 + Nc1[phi_index] + Nc2[phi_index]);

//  printf ("%lf\t%d\t%d\t%d\t%d\t%lf\n", phi1, Nc1[phi_index], Nc2[phi_index], B1[phi_index], B2[phi_index], Gex[phi_index]);
//  printf("%d, %d\n", B1[phi_index], B2[phi_index]);
}

for(phi_index=0; phi_index<11; phi_index++)
{
  printf("%d\n", B1[phi_index]);
}

for(phi_index=0; phi_index<11; phi_index++)
{
  printf("%d\n", B2[phi_index]);
}

} /* end main */

/***********************************************************************************/
/***********************************************************************************/

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
  XFillRectangle(dpy, window, context, 0, 0, WSIZE, WSIZE);
  
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

