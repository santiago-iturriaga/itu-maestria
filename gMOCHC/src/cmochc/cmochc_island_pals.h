#ifndef CMOCHC_ISLANDS_PALS__H_
#define CMOCHC_ISLANDS_PALS__H_

#define VERY_BIG_FLOAT 1073741824

//#define PALS__MAX_INTENTOS 1
//#define PALS__MAX_INTENTOS 5
#define PALS__MAX_INTENTOS 10
//#define PALS__MAX_INTENTOS 100
//#define PALS__MAX_INTENTOS 262144

//#define PALS__SIMPLE_DELTA
#define PALS__COMPLEX_DELTA

#define PALS__SWAP_SEARCH 0.5
#define PALS__MOVE_SEARCH 0.5

#define PALS__MAKESPAN_SEARCH 0.5
#define PALS__ENERGY_SEARCH 0.5

#ifdef DEBUG_1
    extern int CHC_PALS_COUNT_EXECUTIONS[MAX_THREADS];
    extern int CHC_PALS_COUNT_FITNESS_IMPROV[MAX_THREADS];
    extern int CHC_PALS_COUNT_FITNESS_IMPROV_SWAP[MAX_THREADS];
    extern int CHC_PALS_COUNT_FITNESS_IMPROV_MOVE[MAX_THREADS];
    extern int CHC_PALS_COUNT_FITNESS_DECLINE[MAX_THREADS];
    extern int CHC_PALS_COUNT_FITNESS_DECLINE_SWAP[MAX_THREADS];
    extern int CHC_PALS_COUNT_FITNESS_DECLINE_MOVE[MAX_THREADS];    
#endif

void pals_init(int thread_id);
void pals_free(int thread_id);
void pals_search(int thread_id, int solution_index);

#endif
