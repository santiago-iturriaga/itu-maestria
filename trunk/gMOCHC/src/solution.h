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
    
    FLOAT *machine_active_energy_consumption;
    FLOAT *machine_energy_consumption;
    FLOAT energy_consumption;
};

void create_empty_solution(struct solution *new_solution);
void clone_solution(struct solution *dst, struct solution *src);
void free_solution(struct solution *s);

void validate_solution(struct solution *s);
void show_solution(struct solution *s);

inline void recompute_metrics(struct solution *sol) {
    int machines_count = INPUT.machines_count;
   
    FLOAT makespan = 0.0;
    FLOAT *machine_compute_time = sol->machine_compute_time;
    
    for (int machine = 0; machine < machines_count; machine++) {      
        if (makespan < machine_compute_time[machine]) {
            makespan = machine_compute_time[machine];
        }
    }

    sol->makespan = makespan;
    
    FLOAT energy_consumption = 0.0;
    FLOAT *machine_energy_consumption = sol->machine_energy_consumption;
    FLOAT *machine_active_energy_consumption = sol->machine_active_energy_consumption;

    for (int machine = 0; machine < machines_count; machine++) {
        machine_energy_consumption[machine] = machine_active_energy_consumption[machine] + 
            ((makespan - machine_compute_time[machine]) * get_scenario_energy_idle(machine));
        
        energy_consumption += machine_energy_consumption[machine];
    }

    sol->energy_consumption = energy_consumption;
}

inline void recompute_solution(struct solution *sol) {
    int machines_count = INPUT.machines_count;
    int tasks_count = INPUT.tasks_count;
   
    FLOAT makespan = 0.0;
    int *task_assignment = sol->task_assignment;
    FLOAT *machine_compute_time = sol->machine_compute_time;

    for (int machine = 0; machine < machines_count; machine++) {
        machine_compute_time[machine] = 0.0;
    }
    
    for (int task = 0; task < tasks_count; task++) {
        int machine = task_assignment[task];
        machine_compute_time[machine] += get_etc_value(machine, task);
        
        if (makespan < machine_compute_time[machine]) {
            makespan = machine_compute_time[machine];
        }
    }

    sol->makespan = makespan;
    
    FLOAT energy_consumption = 0.0;
    FLOAT *machine_energy_consumption = sol->machine_energy_consumption;
    FLOAT *machine_active_energy_consumption = sol->machine_active_energy_consumption;

    for (int machine = 0; machine < machines_count; machine++) {
        machine_active_energy_consumption[machine] = machine_compute_time[machine] * get_scenario_energy_max(machine);
        machine_energy_consumption[machine] = machine_active_energy_consumption[machine] + 
            ((makespan - machine_compute_time[machine]) * get_scenario_energy_idle(machine));
        
        energy_consumption += machine_energy_consumption[machine];
    }

    sol->energy_consumption = energy_consumption;
    
    #ifdef DEBUG_3
        FLOAT aux;
        aux = 0;
        for (int m = 0; m < INPUT.machines_count; m++) {
            aux += sol->machine_energy_consumption[m];
        }
        assert(sol->energy_consumption == aux);
    #endif
}

#endif
