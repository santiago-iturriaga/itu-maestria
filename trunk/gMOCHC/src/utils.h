#ifndef UTILS_H_
#define UTILS_H_

inline void timming_start(timespec &ts) {
    #if defined(TIMMING)
        clock_gettime(CLOCK_REALTIME, &ts);
    #endif
}

inline void timming_end(char *message, timespec &ts) {
    #if defined(TIMMING)
        timespec ts_end;
        clock_gettime(CLOCK_REALTIME, &ts_end);

        double elapsed;
        elapsed = ((ts_end.tv_sec - ts.tv_sec) * 1000000.0) + ((ts_end.tv_nsec
                - ts.tv_nsec) / 1000.0);
        fprintf(stdout, "[TIMER] %s: %f microsegs.\n", message, elapsed);
    #endif
}

#if defined(TIMMING)
    #define TIMMING_START(ts) timming_start(ts)
    #define TIMMING_END(message,ts) timming_end(message,ts)
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

#endif // UTILS_H_
