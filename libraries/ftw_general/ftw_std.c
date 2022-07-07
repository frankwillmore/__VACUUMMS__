#include "ftw_std.h"

int verbose;

double getDoubleParameter(char *param, char *value)
{
  V printf("%s=%s\n", param, value);
  return strtod(value, NULL);
}

int getIntParameter(char *param, char *value)
{
  V printf("%s=%s\n", param, value);
  return strtol(value, NULL, 10);
}

