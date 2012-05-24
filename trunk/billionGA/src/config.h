//#define INFO
//#define DEBUG
//#define TIMMING

/*#define POPULATION_SIZE     1024
#define DELTA               2
#define MIN_PVALUE          100
#define MAX_PVALUE          924*/

#define POPULATION_SIZE     4096
#define DELTA               1
#define MIN_PVALUE          409
#define MAX_PVALUE          3687

/*#define POPULATION_SIZE     8192
#define DELTA               1
#define MIN_PVALUE          819
#define MAX_PVALUE          7373*/

// The full fitness update model updates the probability vector according
// to the total sample fitness, but requires an extra barrier to sync the threads.
//#define FULL_FITNESS_UPDATE
#define PARTIAL_FITNESS_UPDATE

#define NUMBER_OF_SAMPLES   2
#define SHOW_UPDATE_EVERY   10000

#define INIT_PROB_VECTOR_VALUE  POPULATION_SIZE >> 1
