#include <stdio.h>

#define VACUUMMS_MAJOR_VERSION 1
#define VACUUMMS_MINOR_VERSION 1
#define VACUUMMS_PATCH_VERSION 3

int main()
{
    // put these in a version file to be read
    int major = VACUUMMS_MAJOR_VERSION;
    int minor = VACUUMMS_MINOR_VERSION;
    int patch = VACUUMMS_PATCH_VERSION;
    printf("VACUUMMS version %d.%d.%d\n\n", major, minor, patch);
}
