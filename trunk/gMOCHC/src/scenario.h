#include "config.h"
#include "load_params.h"

#ifndef SCENARIO_H_
#define SCENARIO_H_

struct scenario {
    int machines_count;

    int *cores;
    FLOAT *ssj;
    FLOAT *idle_energy;
    FLOAT *max_energy;
};

void init_scenario(struct params *input, struct scenario *s);
void free_scenario(struct scenario *s);
void show_scenario(struct scenario *s);

void set_scenario_machine(struct scenario *s, int machine, int cores, FLOAT ssj, FLOAT idle_value, FLOAT max_value);

FLOAT get_scenario_ssj(struct scenario *s, int machine);
FLOAT get_scenario_energy_idle(struct scenario *s, int machine);
FLOAT get_scenario_energy_max(struct scenario *s, int machine);
int get_scenario_cores(struct scenario *s, int machine);

#endif /* SCENARIO_H_ */
