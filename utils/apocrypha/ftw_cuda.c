/* ftw_cuda.c */

/* A utility to check status and info on gpu, availability, etc. */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>

#include <ftw_param.h>
#include <ftw_types.h>
#include <ftw_cuda.h>

main(int argc, char *argv[]) 
{
  int device_info=0;
  int device_count=0;

  setCommandLineParameters(argc, argv);

  if (getFlagParam("-usage"))
  {
    printf("usage:     ftw_cuda       -device_info [ 0 ]\n");
    printf("                          -device_count\n");
    exit(0);
  }

  device_count = ftw_getNumberOfDevices();

  if (getFlagParam("-device_info"))
  {
    if (device_count == 0) 
    {
      printf("No device present.\n");
      exit(1);
    }
    getIntParam("-device_info", &device_info);
    printf("Checking device %d...\n", device_info);
    ftw_printDeviceInfo(device_info);
  }

  if (getFlagParam("-device_count")) printf("%d devices found.\n", device_count);
}

