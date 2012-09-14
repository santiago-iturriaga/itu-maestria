#ifndef SCENARIO_STRUCT_H_
#define SCENARIO_STRUCT_H_

#include "config.h"

struct scenario {
    int *cores;
    FLOAT *ssj;
    FLOAT *idle_energy;
    FLOAT *max_energy;
};

#endif /* SCENARIO_STRUCT_H_ */
