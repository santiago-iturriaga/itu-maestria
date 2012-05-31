//#define INFO
//#define DEBUG
//#define TIMMING

/*#define POPULATION_SIZE     768
#define DELTA               64
#define MIN_PVALUE          1
#define MAX_PVALUE          768*/

/*#define POPULATION_SIZE     384
#define DELTA               1
#define MIN_PVALUE          1
#define MAX_PVALUE          385*/

#define POPULATION_SIZE     1024
#define DELTA               1
#define MIN_PVALUE          1
#define MAX_PVALUE          1024

/*#define POPULATION_SIZE     4096
#define DELTA               2
#define MIN_PVALUE          1
#define MAX_PVALUE          4096*/

/*#define POPULATION_SIZE     8192
#define DELTA               1
#define MIN_PVALUE          819
#define MAX_PVALUE          7373*/

// The full fitness update model updates the probability vector according
// to the total sample fitness, but requires an extra barrier to sync the threads.
#define FULL_FITNESS_UPDATE
//#define PARTIAL_FITNESS_UPDATE

#define NUMBER_OF_SAMPLES   2

#define SHOW_UPDATE_EVERY   10000
//#define SHOW_UPDATE_EVERY   10

#define INIT_PROB_VECTOR_VALUE      POPULATION_SIZE >> 1

//#define MACRO_TIMMING

#define NOISE_PROB                  0.005
