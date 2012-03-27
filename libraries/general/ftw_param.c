#include <stdlib.h>
#include <string.h>
#include <stdio.h>

//#include "getParameter.h"
#include <ftw_param.h>

void setCommandLineParameters(int argc, char **argv)
{
  command_line_argc = argc;
  command_line_argv = argv;
}

// usage:       char *test;
//              getStringParam("-test", &test);
void getStringParam(char *param_name, char **parameter)
{
  int i=0;
 
  while (++i<command_line_argc)
  if (strcmp(command_line_argv[i], param_name) == 0) 
  {
    if (i+1>=command_line_argc) 
    {
      printf("no value specified for %s\n", param_name);
      exit(1);
    }

    *parameter = command_line_argv[++i];
  }
}

void getIntParam(char *param_name, int *parameter)
{
  int i=0;
 
  while (++i<command_line_argc)
  if (strcmp(command_line_argv[i], param_name) == 0) 
  {
    if (i+1>=command_line_argc) 
    {
      printf("no value specified for %s\n", param_name);
      exit(1);
    }

    *parameter = (strtol(command_line_argv[++i], NULL, 10));
  }
}

void getLongParam(char *param_name, long *parameter)
{
  int i=0;
 
  while (++i<command_line_argc)
  if (strcmp(command_line_argv[i], param_name) == 0) 
  {
    if (i+1>=command_line_argc) 
    {
      printf("no value specified for %s\n", param_name);
      exit(1);
    }

    *parameter = (strtol(command_line_argv[++i], NULL, 10));
  }
}

void getFloatParam(char *param_name, float *parameter)
{
  int i=0;
 
  while (++i<command_line_argc)
  if (strcmp(command_line_argv[i], param_name) == 0) 
  {
    if (i+1>=command_line_argc) 
    {
      printf("no value specified for %s\n", param_name);
      exit(1);
    }

    *parameter = (float)strtod(command_line_argv[++i], NULL);
  }
}

void getDoubleParam(char *param_name, double *parameter)
{
  int i=0;
 
  while (++i<command_line_argc)
  if (strcmp(command_line_argv[i], param_name) == 0) 
  {
    if (i+1>=command_line_argc) 
    {
      printf("no value specified for %s\n", param_name);
      exit(1);
    }

    *parameter = (strtod(command_line_argv[++i], NULL));
  }
}

void getVectorParam(char *param_name, double *parameter1,  double *parameter2, double *parameter3)
{
  int i=0;
 
  while (++i<command_line_argc)
  if (strcmp(command_line_argv[i], param_name) == 0) 
  {
    if (i+3>=command_line_argc) 
    {
      printf("not enough values specified for %s\n", param_name);
      exit(1);
    }

    *parameter1 = (strtod(command_line_argv[++i], NULL));
    *parameter2 = (strtod(command_line_argv[++i], NULL));
    *parameter3 = (strtod(command_line_argv[++i], NULL));
  }
}

void getVectorStringParam(char *param_name, char **parameter1,  char **parameter2, char **parameter3)
{
  int i=0;
 
  while (++i<command_line_argc)
  if (strcmp(command_line_argv[i], param_name) == 0) 
  {
    if (i+3>=command_line_argc) 
    {
      printf("not enough values specified for %s\n", param_name);
      exit(1);
    }

    *parameter1 = command_line_argv[++i];
    *parameter2 = command_line_argv[++i];
    *parameter3 = command_line_argv[++i];
  }
}

int getFlagParam(char *param_name)
{
  int i=0;

  while (++i<command_line_argc) if (strcmp(command_line_argv[i], param_name) == 0) return 1;

  return 0;
}

