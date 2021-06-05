#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <sys/time.h>

#include "utils.h"

#ifndef MAT_VAL_TYPE
#define MAT_VAL_TYPE double
#endif

#ifndef MAT_PTR_TYPE
#define MAT_PTR_TYPE int
#endif

//#ifndef SubNum
//#define SubNum  3
//#endif


#define WARP_SIZE 32
#define WARP_PER_BLOCK 4

#define HALFWARP_SIZE WARP_SIZE/2
#define HALFWARP_PER_BLOCK 2*WARP_PER_BLOCK

#ifndef BLOCK_SIZE
#define BLOCK_SIZE  16
#endif

#ifndef NTHREADS_MAX
#define NTHREADS_MAX 1
#endif


#define METHOD_SERIAL               1
#define METHOD_LOCK                 2
#define METHOD_LOCK_AND_DEPENDENCY  3

#define HASH_SCALE 107
