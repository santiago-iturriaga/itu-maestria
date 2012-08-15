#ifndef UTILS_H_
#define UTILS_H_

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>

#include "config.h"

inline void timming_start(timespec &ts) {
    #if defined(TIMMING)
        clock_gettime(CLOCK_REALTIME, &ts);
    #endif
}

inline void timming_end(const char *message, timespec &ts) {
    #if defined(TIMMING)
        timespec ts_end;
        clock_gettime(CLOCK_REALTIME, &ts_end);

        double elapsed;
        elapsed = ((ts_end.tv_sec - ts.tv_sec) * 1000000.0) + ((ts_end.tv_nsec
                - ts.tv_nsec) / 1000.0);
        fprintf(stdout, "[TIMER] %s: %f microsegs.\n", message, elapsed);
    #endif
}

#if defined(DEBUG_LEVEL)
    #define ASSERT(a) assert(a);
#else
    #define ASSERT(a)
#endif

#endif // UTILS_H_
