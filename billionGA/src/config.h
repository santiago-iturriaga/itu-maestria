//#define INFO
//#define DEBUG
//#define TIMMING

//#define POPULATION_SIZE   512
#define POPULATION_SIZE   1024
//#define POPULATION_SIZE   4096

#define DELTA           2
#define MIN_PVALUE      100
#define MAX_PVALUE      924

//#define CPU_LOOP
#define GPU_LOOP

// The full fitness update model updates the probability vector according
// to the total sample fitness, but requires an extra barrier to sync the threads.
//#define FULL_FITNESS_UPDATE
#define PARTIAL_FITNESS_UPDATE

#define NUMBER_OF_SAMPLES   2

#define SHOW_UPDATE_EVERY   1

#define INIT_PROB_VECTOR_VALUE  POPULATION_SIZE >> 1
