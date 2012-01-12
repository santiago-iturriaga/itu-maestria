#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "utils.h"
#include "config.h"

void timming_start(timespec &ts) {
	/*if (TIMMING) {
		clock_gettime(CLOCK_REALTIME, &ts);
	}*/
}

void timming_end(char *message, timespec &ts) {
	/*if (TIMMING) {
		timespec ts_end;
		clock_gettime(CLOCK_REALTIME, &ts_end);

		double elapsed;
		elapsed = ((ts_end.tv_sec - ts.tv_sec) * 1000000.0) + ((ts_end.tv_nsec
				- ts.tv_nsec) / 1000.0);
		fprintf(stdout, "[TIMMING] %s: %f microsegs.\n", message, elapsed);
	}*/
}
