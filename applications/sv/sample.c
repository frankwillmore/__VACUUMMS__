/* sample.c */

#include <ftw_std.h>
#include <ftw_rng2.h>
#include <ftw_param.h>

main(int argc, char* argv[])
{
  setCommandLineParameters(argc, argv);
  if (getFlagParam("-randomize")) initializeRandomNumberGenerator2(-1);
  else initializeRandomNumberGenerator2(0);

  printf("%lf\n", rnd2());
}
