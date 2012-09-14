#ifndef ETC_MATRIX_H_
#define ETC_MATRIX_H_

#include <assert.h>

#include "config.h"
#include "global.h"
#include "etc_matrix_struct.h"

void init_etc_matrix();
void free_etc_matrix();
void show_etc_matrix();

void set_etc_value(int machine, int task, FLOAT value);

inline int get_etc_coord(int machine, int task) {
    assert(machine < INPUT.machines_count);
    assert(machine >= 0);
    assert(task < INPUT.tasks_count);
    assert(task >= 0);

    return ETC.data_machine_index[machine] + task;
}

inline FLOAT get_etc_value(int machine, int task) {
    return ETC.data[get_etc_coord(machine, task)];
}

#endif /* ETC_MATRIX_H_ */
