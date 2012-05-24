#include <time.h>

#ifndef UTILS_H_
#define UTILS_H_

void timming_start(timespec &ts);
void timming_end(char *message, timespec &ts);

#endif // UTILS_H_
