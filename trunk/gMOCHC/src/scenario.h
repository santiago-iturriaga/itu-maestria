#ifndef SCENARIO_H_
#define SCENARIO_H_

#include "config.h"
#include "global.h"
#include "scenario_struct.h"

void init_scenario();
void free_scenario();
void show_scenario();

void set_scenario_machine(int machine, int cores, FLOAT ssj, FLOAT idle_value, FLOAT max_value);

inline int get_scenario_cores(int machine) {
    return SCENARIO.cores[machine];
}

inline FLOAT get_scenario_energy_idle(int machine) {
    return SCENARIO.idle_energy[machine];
}

inline FLOAT get_scenario_energy_max(int machine) {
    return SCENARIO.max_energy[machine];
}

inline FLOAT get_scenario_ssj(int machine) {
    return SCENARIO.ssj[machine];
}

#endif /* SCENARIO_H_ */
