#define INFO
#define DEBUG

//#define POPULATION_SIZE sqrt(3.1416)*sqrt(number_of_bits)*log10(number_of_bits)/2
//#define POPULATION_SIZE 500
#define POPULATION_SIZE 100

// The full fitness update model updates the probability vector according
// to the total sample fitness, but requires an extra barrier to sync the threads.
//#define FULL_FITNESS_UPDATE

#define PARTIAL_FITNESS_UPDATE

#define NUMBER_OF_SAMPLES       2

// Debe ser divisible entre 32 (8 y 4)
//#define MAX_PROB_VECTOR_BITS    1048576
//#define MAX_PROB_VECTOR_BITS    536870912
//#define MAX_PROB_VECTOR_BITS    899999744
//#define MAX_PROB_VECTOR_BITS    1073741824

#define INIT_PROB_VECTOR_VALUE  0.5

//#define RNUMBERS_PER_GEN        64
//#define RNUMBERS_PER_GEN        84
//#define RNUMBERS_PER_GEN        168
//#define RNUMBERS_PER_GEN        1000020
//#define RNUMBERS_PER_GEN        1048576
//#define RNUMBERS_PER_GEN        67108864
//#define RNUMBERS_PER_GEN        10000032
//#define RNUMBERS_PER_GEN        134217728
