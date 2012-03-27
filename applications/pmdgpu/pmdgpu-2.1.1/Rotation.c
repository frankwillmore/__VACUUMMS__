/* Rotation.c */

#include <pmdgpu.h>

/************************************************************************/
/*                                                                      */
/* The functions here use a quaternion formulation to rotate an input   */
/* vector about the input axis.  In setAxisOfRotation, an axis is       */
/* selected at random from the set of axes that are perpendicular to    */
/* the axis of the molecule.   Thus, the molecule is not moved, but the */
/* direction of rotation is thus changed.                               */
/*                                                                      */
/************************************************************************/

// IN:  (Unit) Axis of molecule(rx, ry, rz), Random number generator to use
// OUT: (Unit) Random axis (px, py, pz) perpendicular to axis of molecule
   
void setAxisOfRotation(float rx, float ry, float rz, struct MersenneTwister *rng, float *px, float *py, float *pz)
{
//printf("molecule axis:  %f\t%f\t%f\n", rx, ry, rz);
  float u = ry * ry / (rx * rx);
  float angle = 2.0f * PI * rndm(rng);
  
  // use intersection of {set of all vectors perpendicular to bond axis} with x-y plane 
  // to get one vector perpendicular to the bond axis.
  *px = sqrt(u / (1 + u));
  *py = sqrt(1 / (1 + u));
  *pz = 0.0f; 
//printf("unrotated axis:  %f\t%f\t%f\n", *px, *py, *pz);
//printf("rotation angle:  %f\n", angle);

  // Rotate this vector about the bond axis through an arbitrary angle [0..2PI].
  quaternionRotate(*px, *py, *pz, rx, ry, rz, angle, px, py, pz);
//printf("rotated axis:  %f\t%f\t%f\n", *px, *py, *pz);
}

// IN:  Vector (vx, vy, vz), (unit) Axis of Rotation, Angle of Rotation
// OUT: Rotated Vector (vxnew, vynew, vznew) 

void quaternionRotate(float vx, float vy, float vz, float axisx, float axisy, float axisz, float angle, float *vxnew, float *vynew, float *vznew)
{
  float a, b, c, d; // components of quaternion z;
  float t2, t3, t4, t5, t6, t7, t8, t9, t10;  // temporary variables;
  float half_angle = angle * 0.5f;
  float sin_half_angle = sin(half_angle);

  a = cos(half_angle);
  b = axisx * sin_half_angle;
  c = axisy * sin_half_angle;
  d = axisz * sin_half_angle;

  t2 =   a*b;
  t3 =   a*c;
  t4 =   a*d;
  t5 =  -b*b;
  t6 =   b*c;
  t7 =   b*d;
  t8 =  -c*c;
  t9 =   c*d;
  t10 = -d*d;

 *vxnew = 2 * ((t8 + t10) * vx + (t6 -  t4) * vy + (t3 + t7) * vz) + vx;
 *vynew = 2 * ((t4 +  t6) * vx + (t5 + t10) * vy + (t9 - t2) * vz) + vy;
 *vznew = 2 * ((t7 -  t3) * vx + (t2 +  t9) * vy + (t5 + t8) * vz) + vz;
}

