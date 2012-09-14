/*
 * Handles a solution in memory.
 */

#ifndef SOLUTION_H_
#define SOLUTION_H_

#include <stdio.h>

#include "config.h"
#include "scenario.h"
#include "etc_matrix.h"
#include "energy_matrix.h"
#include "global.h"

#define SOLUTION__NOT_INITIALIZED -1
#define SOLUTION__EMPTY 0
#define SOLUTION__IN_USE 1
#define SOLUTION__TASK_NOT_ASSIGNED -1

struct solution {
    int initialized;
    
    int *task_assignment;
    
    FLOAT *machine_compute_time;
    FLOAT makespan;
    
    FLOAT *machine_energy_consumption;
    FLOAT energy_consumption;
};

void create_empty_solution(struct solution *new_solution);
void clone_solution(struct solution *dst, struct solution *src);
void free_solution(struct solution *s);

void validate_solution(struct solution *s);
void show_solution(struct solution *s);

inline void refresh_solution(struct solution *sol) {
    for (int machine = 0; machine < INPUT.machines_count; machine++) {
        sol->machine_compute_time[machine] = 0.0;
    }

    sol->makespan = 0.0;

    for (int task = 0; task < INPUT.tasks_count; task++) {
        int machine = sol->task_assignment[task];

        sol->machine_compute_time[machine] += get_etc_value(machine, task);
        if (sol->makespan < sol->machine_compute_time[machine]) {
            sol->makespan = sol->machine_compute_time[machine];
        }
    }

    sol->energy_consumption = 0.0;

    for (int machine = 0; machine < INPUT.machines_count; machine++) {
        sol->machine_energy_consumption[machine] = sol->makespan * get_scenario_energy_idle(machine);
        sol->energy_consumption += sol->machine_energy_consumption[machine];
    }

    FLOAT task_energy_consumption;
    for (int task = 0; task < INPUT.tasks_count; task++) {
        int machine = sol->task_assignment[task];

        task_energy_consumption = get_energy_value(machine, task);

        sol->machine_energy_consumption[machine] += task_energy_consumption;
        sol->energy_consumption += task_energy_consumption;
    }
}

#endif
