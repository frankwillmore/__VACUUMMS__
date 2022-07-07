/* ftw_param.h */

#ifndef FTW_PARAM_H
#define FTW_PARAM_H

#include <stdlib.h>

/* FTW: symbols below moved to the source file */
/* int command_line_argc;
/* char **command_line_argv;
 */

void setCommandLineParameters(int argc, char **argv);

/* if a parameter is received, return a true value, otherwise return NULL) */
int getIntParam(char *param_name, int *parameter);
int getLongParam(char *param_name, long *parameter);
int getFloatParam(char *param_name, float *parameter);
int getDoubleParam(char *param_name, double *parameter);
int getStringParam(char *param_name, char **parameter);
//void getStringParam(char *param_name, char *parameter);
int getVectorParam(char *param_name, double *parameter1,  double *parameter2, double *parameter3);
int getVectorStringParam(char *param_name, char **parameter1,  char **parameter2, char **parameter3);
int getFlagParam(char *param_name);


// usage: 	char *test;
// 		getStringParam("-test", &test);

#endif
