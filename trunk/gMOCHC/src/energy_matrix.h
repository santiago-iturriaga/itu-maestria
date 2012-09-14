#ifndef ENERGY_MATRIX_H_
#define ENERGY_MATRIX_H_

#include <assert.h>

#include "energy_matrix_struct.h"

#include "config.h"
#include "global.h"

void init_energy_matrix();
void free_energy_matrix();

void set_energy_value(int machine, int task, FLOAT value);

inline int get_energy_coord(int machine, int task) {
    assert(machine < INPUT.machines_count);
    assert(machine >= 0);
    assert(task < INPUT.tasks_count);
    assert(task >= 0);

    return ENERGY.data_machine_index[machine] + task;
}

inline FLOAT get_energy_value(int machine, int task) {
    return ENERGY.data[get_energy_coord(machine, task)];
}

#endif /* ENERGY_MATRIX_H_ */
