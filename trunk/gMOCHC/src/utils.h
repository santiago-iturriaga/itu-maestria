#ifndef UTILS_H_
#define UTILS_H_

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <string.h>

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

        FLOAT elapsed;
        elapsed = ((ts_end.tv_sec - ts.tv_sec) * 1000000.0) + ((ts_end.tv_nsec
                - ts.tv_nsec) / 1000.0);
        fprintf(stderr, "[TIMER] %s: %f microsegs.\n", message, elapsed);
    #endif
}

inline char* int_to_binary(int x) {
    //fprintf(stderr, ">> x=%d\n", x);

    char *b = (char*)(malloc(sizeof(char) * 33));
    b[32] = '\0';

    int mask = 0x1;

    int z;
    for (z = 31; z >= 0; z--) {
        //fprintf(stderr, ">> z=%d mask=%d\n", z, mask);
        
        b[z] = ((x & mask) != 0) ? '1' : '0';
        mask = mask << 1;
    }

    return b;
}

#endif // UTILS_H_
