#include "cpu_rand.h"
#include "cpu_drand48.h"
#include "cpu_mt.h"

/* RAND_GENERATE should return nonnegative double-precision 
 * floating-point values uniformly distributed
 * between [0.0, 1.0). */

#if defined(CPU_RAND)
    #define RAND_STATE struct cpu_rand_state
    #define RAND_INIT(seed,state) cpu_rand_init(seed,state);
    #define RAND_GENERATE(state) cpu_rand_generate(state);
    #define RAND_FINALIZE(state) cpu_rand_free(state);
#endif

#if defined(CPU_DRAND48)
    #define RAND_STATE struct cpu_drand48_state
    #define RAND_INIT(seed,state) cpu_drand48_init(seed,ts);
    #define RAND_GENERATE(state) cpu_drand48_generate(state);
    #define RAND_FINALIZE(state) cpu_drand48_free(state);
#endif

#if defined(CPU_MT)
    #define RAND_STATE struct cpu_mt_state
    #define RAND_INIT(seed,state) cpu_mt_init(seed,state);
    #define RAND_GENERATE(state) cpu_mt_generate(state);
    #define RAND_FINALIZE(state) cpu_mt_free(state);
#endif
