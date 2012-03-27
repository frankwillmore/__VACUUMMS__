/* ThreadMain.cu */

/******************************************************************************/
/*                                                                            */
/* The content of ThreadMain() are essentially the same as main() in the      */
/* original program utrwt69.  The particle is inserted and moves forward in   */
/* a random direction until hitting a collision point.  Repeat until the      */
/* target time of the simulation has passed.  Since the time of the particle  */
/* motion is not monitored directly, the midpoint of each segment is used to  */
/* calculate a weighted average of the position of the particle vs time. The  */
/* weighting factor used is the amount of time spent on each segment of the   */
/* trajectory.                                                                */
/*                                                                            */
/* The factor of 8.314 in the kinetic_energy term is used to account for the  */
/* epsilon values provided which are divided by Boltzmann's constant.  The    */
/* values are placed on a per-mole basis by dividing by the gas constant      */
/* R = 8.314 J/mol K.                                                         */ 
/*                                                                            */
/******************************************************************************/

#include <pmdgpu.h>

extern struct CommandLineOptions clo;

// Use mutex to synchronize access to time series data
extern sem_t thread_sem;
extern pthread_mutex_t time_series_mutex;
extern float time_series_R_delta_t[];
extern float time_series_Rsq_delta_t[];
extern float time_series_delta_t[];

void *ThreadMain(void *passval)
{
  struct ThreadResult *thread;
  thread = (struct ThreadResult*) passval;

  // Wait for the signal, then start the thread.
  sem_wait(&thread_sem);

  /* create and initialize random number sequence for thread */
  thread->rng = (struct MersenneTwister*)malloc(sizeof(struct MersenneTwister));
  MersenneInitialize(thread->rng, clo.seed+thread->thread_id);

  /* set start point and initialize variables */
  thread->drift_x=0;
  thread->drift_y=0;
  thread->drift_z=0;
  thread->time_elapsed=0;
  generateTestPoint(thread);
  calculateTestpointEnergy(thread);

  /***********************************/
  /*                                 */
  /*     begin loop over segments    */
  /*                                 */
  /***********************************/

  while (thread->time_elapsed < clo.target_time)
  {
    thread->collision_x = thread->drift_x + thread->test_x;
    thread->collision_y = thread->drift_y + thread->test_y;
    thread->collision_z = thread->drift_z + thread->test_z;
    thread->collision_t = thread->time_elapsed;
    setVelocitySpeed(thread);
    setRotationalOrientation(thread);
    setRotationalEnergy(thread);
    
    // keep choosing a velocity direction until we have one that works
    int condition_A = 0;
    int condition_B = 0;
    int visit_count = 0; // get it unstuck
    while (!condition_A && !condition_B && visit_count++ < 15)
    {
      setVelocityDirection(thread);
      calculateEnergyArray(thread);
      condition_A = (thread->energy_array[0] <= (thread->translational_kinetic_energy + thread->rotational_kinetic_energy)/8.314); // fast enough
      condition_B = (thread->energy_array[0] <= thread->testpoint_energy); // rolling downhill in energy
    }
    
    // extend particle path forward from collision point
    int energy_index=0;
    while (condition_A || condition_B)
    {
        thread->test_x+=(thread->resolution_x);
        thread->test_y+=(thread->resolution_y);
        thread->test_z+=(thread->resolution_z);
        thread->testpoint_energy = thread->energy_array[energy_index];

        if (energy_index == 31) // keep moving forward at this resolution
        {
          makeVerletList(thread);
          calculateEnergyArray(thread);
          energy_index = -1;
        }
        energy_index++;
        condition_A = (thread->energy_array[energy_index] <= (thread->translational_kinetic_energy + thread->rotational_kinetic_energy)/8.314); // fast enough
        condition_B = (thread->energy_array[energy_index] <= thread->testpoint_energy); // rolling downhill in energy
    }
    
    // now refine the collision point

    thread->resolution_x *= thirtysecondth;
    thread->resolution_y *= thirtysecondth;
    thread->resolution_z *= thirtysecondth;

    calculateTestpointEnergy(thread);
    calculateEnergyArray(thread);

    for (energy_index=0; energy_index<32; energy_index++)
    {
      condition_A = (thread->energy_array[energy_index] <= (thread->translational_kinetic_energy + thread->rotational_kinetic_energy)/8.314); // fast enough
      condition_B = (thread->energy_array[energy_index] <= thread->testpoint_energy); // rolling downhill in energy
      if (condition_A || condition_B)
      {
        thread->test_x+=(thread->resolution_x);
        thread->test_y+=(thread->resolution_y);
        thread->test_z+=(thread->resolution_z);
        thread->testpoint_energy = thread->energy_array[energy_index];
      }
      else 
      {
        break; // found the point
      }
    } 

    /*************/
    /*           */
    /*   CLOCK   */
    /*           */
    /*************/
 
    float segment_dx = thread->drift_x + thread->test_x - thread->collision_x;
    float segment_dy = thread->drift_y + thread->test_y - thread->collision_y;
    float segment_dz = thread->drift_z + thread->test_z - thread->collision_z;
    float segment_length = sqrt(segment_dx * segment_dx + segment_dy * segment_dy + segment_dz * segment_dz);
    float segment_dt = segment_length / thread->speed;

    // midpoint is relative to (test_x0, test_y0, test_z0), so add drift vals
    float mid_x = thread->collision_x + 0.5 * segment_dx;
    float mid_y = thread->collision_y + 0.5 * segment_dy;
    float mid_z = thread->collision_z + 0.5 * segment_dz;

    float mid_t = thread->time_elapsed + 0.5 * segment_dt;

    int t_bin = (int)floor((mid_t / clo.resolution_t) + 0.5);
    float Rsq_t = (mid_x - thread->test_x0) * (mid_x - thread->test_x0)
                + (mid_y - thread->test_y0) * (mid_y - thread->test_y0)
                + (mid_z - thread->test_z0) * (mid_z - thread->test_z0);
  
    // synchronized code...
    pthread_mutex_lock( &time_series_mutex );
      time_series_Rsq_delta_t[t_bin] += Rsq_t * segment_dt;
      time_series_delta_t[t_bin] += segment_dt;
    pthread_mutex_unlock( &time_series_mutex );

    thread->time_elapsed += segment_dt;
 
  }  // loop until time's up!

  // Free memory, alert the next thread that it can run, then exit
  free(thread->rng);
  sem_post(&thread_sem);
  pthread_exit(NULL);
}

void generateTestPoint(struct ThreadResult *thread)
{
  thread->test_x = thread->test_x0 = rndm(thread->rng) * clo.box_x;
  thread->test_y = thread->test_y0 = rndm(thread->rng) * clo.box_y;
  thread->test_z = thread->test_z0 = rndm(thread->rng) * clo.box_z;

  makeVerletList(thread);
  setRotationalOrientation(thread);
  calculateTestpointEnergy(thread);

  while (rndm(thread->rng) > exp(-(thread->testpoint_energy/clo.T)))
  {
    thread->test_x = thread->test_x0 = rndm(thread->rng) * clo.box_x;
    thread->test_y = thread->test_y0 = rndm(thread->rng) * clo.box_y;
    thread->test_z = thread->test_z0 = rndm(thread->rng) * clo.box_z;
  
    makeVerletList(thread);
    setRotationalOrientation(thread);  
    calculateTestpointEnergy(thread);
  }
}

void setVelocityDirection(struct ThreadResult *thread)
{
  // pick a direction
  thread->translation_phi = 2*PI*rndm(thread->rng);
  // sample acos to generate theta 0 (0..PI) as sinusoidal distribution
  thread->translation_theta = acos(1.0f - 2.0f * rndm(thread->rng));
  thread->resolution_x = thirtysecondth * cos(thread->translation_phi)*sin(thread->translation_theta);
  thread->resolution_y = thirtysecondth * sin(thread->translation_phi)*sin(thread->translation_theta);
  thread->resolution_z = thirtysecondth * cos(thread->translation_theta);
}

void setVelocitySpeed(struct ThreadResult *thread)
{
  // pick a speed
  thread->translational_kinetic_energy = -1.5*log(rndm(thread->rng))*8.314*clo.T; // in J/mol
  thread->speed = sqrt(2.0*thread->translational_kinetic_energy/clo.test_mass) * 0.01; // 0.01 to go from m/s to A/ps
}

void setRotationalOrientation(struct ThreadResult *thread)
{
  thread->orientation_phi = 2 * PI * rndm(thread->rng);
  thread->orientation_theta = acos(1.0f - 2.0f * rndm(thread->rng));
  thread->orientation_dx1 = clo.bond_length1 * cos(thread->orientation_phi) * sin(thread->orientation_theta);
  thread->orientation_dy1 = clo.bond_length1 * sin(thread->orientation_phi) * sin(thread->orientation_theta);
  thread->orientation_dz1 = clo.bond_length1 * cos(thread->orientation_theta);
  thread->orientation_dx2 = clo.bond_length2 * cos(thread->orientation_phi) * sin(thread->orientation_theta);
  thread->orientation_dy2 = clo.bond_length2 * sin(thread->orientation_phi) * sin(thread->orientation_theta);
  thread->orientation_dz2 = clo.bond_length2 * cos(thread->orientation_theta);
}

void setRotationalEnergy(struct ThreadResult *thread)
{
  thread->rotational_kinetic_energy = 0; // -1.0f * log(rndm(thread->rng)) * 8.314 * clo.T;  
}
