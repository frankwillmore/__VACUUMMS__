/* maltest.cu */

#include <ftw_config_parser.h>

#include <ftw_types.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>

main(int argc, char *argv[]) 
{
  int device_count;
  cudaError_t err = cudaErrorUnknown;

//  REMOVING either device count or the readGFG ALLOWS THE MALLOC TO WORK.  WTF?
  cudaGetDeviceCount(&device_count);
  printf("%d device found.\n", device_count);

err=cudaGetLastError();
fprintf(stderr, "main::%s\n", cudaGetErrorString(err));

  ftw_GFG65536 *gfg = readGFG65536(stdin);


  int *p_int;

fprintf(stderr, "main::%s\n", cudaGetErrorString(err));
fprintf(stderr, "main::p_int:  %p\n", p_int);
err = cudaMalloc( (void **) &p_int, 65536 * sizeof(int));
fprintf(stderr, "main::p_int:  %p\n", p_int);
fprintf(stderr, "main::%s\n", cudaGetErrorString(err));

sleep(1);

cudaSetDevice(1);

err = cudaMalloc( (void **) &p_int, 65536 * sizeof(int));
fprintf(stderr, "main::%s\n", cudaGetErrorString(err));
}


