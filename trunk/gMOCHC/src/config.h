#ifndef CONFIG_H_
#define CONFIG_H_

/* Outputs solution(s) to stdout */
#define OUTPUT_SOLUTION

/* Records the execution time of 
 * some important operations */
//#define TIMMING

/* Random number generator */
//#define CPU_RAND
#define CPU_DRAND48
//#define CPU_MT

/* Floating point precision */
//#define FLOAT float
//#define DISPLAY_PRECISION "float"

#define FLOAT double
#define DISPLAY_PRECISION "double"

/* Max. supported number of threads */
#define MAX_THREADS 64

/* Debug level
 * 0 No debug
 * 1 Minimal debug
 * 2 Medium debug
 * 3 Debug everything */
#define DEBUG_LEVEL 1

//#define REPORT_EVERY_SECONDS 1
//#define REPORT_EVERY_ITERS   1

#if defined(TIMMING)
    #define TIMMING_START(ts) timespec ts;timming_start(ts);
    #define TIMMING_END(message,ts) timming_end(message,ts);
#else
    #define TIMMING_START(ts)
    #define TIMMING_END(message,ts)
#endif

#if defined(DEBUG_LEVEL)
    #define DEBUG_0
#endif

#if defined(DEBUG_LEVEL) && DEBUG_LEVEL >= 1
    #define DEBUG_1
#endif

#if defined(DEBUG_LEVEL) && DEBUG_LEVEL >= 2
    #define DEBUG_2
#endif

#if defined(DEBUG_LEVEL) && DEBUG_LEVEL >= 3
    #define DEBUG_3
#endif

#endif //CONFIG_H_
