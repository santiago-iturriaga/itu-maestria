#ifndef CMOCHC_ISLANDS_PALS__H_
#define CMOCHC_ISLANDS_PALS__H_

#define PALS__MAX_INTENTOS 50

#ifdef DEBUG_3
    extern int CHC_PALS_COUNT_EXECUTIONS[MAX_THREADS];
    extern int CHC_PALS_COUNT_FITNESS_IMPROV[MAX_THREADS];
    extern int CHC_PALS_COUNT_FITNESS_DECLINE[MAX_THREADS];
#endif

void pals_init(int thread_id);
void pals_free(int thread_id);
void pals_search(int thread_id, int solution_index);

#endif
