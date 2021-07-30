/* ftw_cav_parser.h */

#ifndef FTW_CAV_PARSER_H
#define FTW_CAV_PARSER_H

//  A set of library routines to parse .cav input file

//  IN:  FILE* pointer to an input stream
//  OUT:  configuration, and metadata:  number of lines

#include <stdlib.h>
#include <stdio.h>
#include <ftw_param.h>
#include <string.h>

#include <ftw_types.h>

// return value is pointer to configuration

#ifdef __cplusplus
extern "C"
#endif
ftw_CAV65536 *readCAV65536(FILE *instream);

#ifdef __cplusplus
extern "C"
#endif
ftw_CAV65536 *replicateCAV65536(ftw_CAV65536 *in);

#endif

