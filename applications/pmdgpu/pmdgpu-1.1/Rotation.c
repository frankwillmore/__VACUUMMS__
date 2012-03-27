/* Rotation.c */

#include <pmdgpu.h>

// IN:  (Unit) Axis of molecule, Random number generator to use
// OUT: (Unit) Random axis perpendicular to axis of molecule
   
;
void getAxisOfRotation(float r1, float r2, float r3, struct MersenneTwister *rng, float *p1, float *p2, float *p3)
{
  float u = r2 * r2 / (r1 * r1);
  float angle = 2.0f * PI * rndm(rng);
  
  // use intersection of {set of all vectors perpendicular to bond axis} with x-y plane 
  // to get one vector perpendicular to the bond axis.
  *p1 = sqrt(u / (1 + u));
  *p2 = sqrt(u / (1 + u));
  *p3 = 0.0f; 

  // Rotate this vector about the bond axis through an arbitrary angle [0..2PI].
  quaternionRotate(*p1, *p2, *p3, r1, r2, r3, angle, p1, p2, p3);
}

// IN:  Vector, (Unit) Axis of Rotation, Angle of Rotation
// OUT: Rotated Vector 
void quaternionRotate(float v1, float v2, float v3, float axis1, float axis2, float axis3, float angle, float *v1new, float *v2new, float *v3new)
{
  float a, b, c, d; // components of quaternion z;
  float t2, t3, t4, t5, t6, t7, t8, t9, t10;  // temporary variables;
  float half_angle = angle * 0.5f;
  float sin_half_angle = sin(half_angle);

  a = cos(half_angle);
  b = axis1 * sin_half_angle;
  c = axis2 * sin_half_angle;
  d = axis3 * sin_half_angle;

  t2 =   a*b;
  t3 =   a*c;
  t4 =   a*d;
  t5 =  -b*b;
  t6 =   b*c;
  t7 =   b*d;
  t8 =  -c*c;
  t9 =   c*d;
  t10 = -d*d;

 *v1new = 2 * ((t8 + t10) * v1 + (t6 -  t4) * v2 + (t3 + t7) * v3) + v1;
 *v2new = 2 * ((t4 +  t6) * v1 + (t5 + t10) * v2 + (t9 - t2) * v3) + v2;
 *v3new = 2 * ((t7 -  t3) * v1 + (t2 +  t9) * v2 + (t5 + t8) * v3) + v3;
}

