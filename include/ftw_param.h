/* ftw_param.h */

#ifndef FTW_PARAM_H
#define FTW_PARAM_H

#include <stdlib.h>

int command_line_argc;
char **command_line_argv;

void setCommandLineParameters(int argc, char **argv);
void getIntParam(char *param_name, int *parameter);
void getLongParam(char *param_name, long *parameter);
void getFloatParam(char *param_name, float *parameter);
void getDoubleParam(char *param_name, double *parameter);
void getStringParam(char *param_name, char **parameter);
//void getStringParam(char *param_name, char *parameter);
void getVectorParam(char *param_name, double *parameter1,  double *parameter2, double *parameter3);
void getVectorStringParam(char *param_name, char **parameter1,  char **parameter2, char **parameter3);
int getFlagParam(char *param_name);


// usage: 	char *test;
// 		getStringParam("-test", &test);

#endif
