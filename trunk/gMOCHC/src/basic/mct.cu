#include "mct.h"

#include "../config.h"
#include "../global.h"

void compute_mct(struct solution *sol) {
    #if defined(DEBUG_3)
        fprintf(stderr, "[DEBUG] calculando MCT...\n");
    #endif
    
    sol->makespan = 0.0;

    int tasks_count = INPUT.tasks_count;
    int machines_count = INPUT.machines_count;
    
    FLOAT *machine_compute_time = sol->machine_compute_time;
    FLOAT *machine_active_energy_consumption = sol->machine_active_energy_consumption;
    int *task_assignment = sol->task_assignment;

    for (int task = 0; task < tasks_count; task++) {
        int best_machine;
        best_machine = 0;

        FLOAT best_etc_value;
        best_etc_value = machine_compute_time[best_machine] 
            + get_etc_value(best_machine, task);

        for (int machine = 1; machine < machines_count; machine++) {
            FLOAT etc_value;
            etc_value = machine_compute_time[machine] 
                + get_etc_value(machine, task);

            if (etc_value < best_etc_value) {
                best_etc_value = etc_value;
                best_machine = machine;
            }
        }

        task_assignment[task] = best_machine;
        machine_compute_time[best_machine] = best_etc_value;
    }
    
    for (int machine = 0; machine < machines_count; machine++) {
        machine_active_energy_consumption[machine] = machine_compute_time[machine] * get_scenario_energy_max(machine);
    }
    
    recompute_metrics(sol);
    sol->initialized = SOLUTION__IN_USE;
    
    #if defined(DEBUG_3)
        fprintf(stderr, "[DEBUG] MCT solution: makespan=%f energy=%f.\n", sol->makespan, sol->energy_consumption);
    #endif
}

void compute_mct_random(struct solution *sol, int start, int direction) {
    #if defined(DEBUG_3)
        fprintf(stderr, "[DEBUG] calculando random MCT...\n");
    #endif
    
    sol->makespan = 0.0;

    int tasks_count = INPUT.tasks_count;
    int machines_count = INPUT.machines_count;
    
    FLOAT *machine_compute_time = sol->machine_compute_time;
    FLOAT *machine_active_energy_consumption = sol->machine_active_energy_consumption;
    int *task_assignment = sol->task_assignment;

    start = start % tasks_count;
    direction = direction % 2;

    int task;
    for (int offset = 0; offset < tasks_count; offset++) {
        if (direction == 0) {
            task = (start + offset) % tasks_count;
        } else {
            task = (start - offset);
            if (task < 0) task = tasks_count + task;
        }
        
        int best_machine;
        best_machine = 0;

        FLOAT best_etc_value;
        best_etc_value = machine_compute_time[best_machine] 
            + get_etc_value(best_machine, task);

        for (int machine = 1; machine < machines_count; machine++) {
            FLOAT etc_value;
            etc_value = machine_compute_time[machine] 
                + get_etc_value(machine, task);

            if (etc_value < best_etc_value) {
                best_etc_value = etc_value;
                best_machine = machine;
            }
        }

        task_assignment[task] = best_machine;
        machine_compute_time[best_machine] = best_etc_value;
    }
    
    for (int machine = 0; machine < machines_count; machine++) {
        machine_active_energy_consumption[machine] = machine_compute_time[machine] * get_scenario_energy_max(machine);
    }
    
    recompute_metrics(sol);
    sol->initialized = SOLUTION__IN_USE;

    #if defined(DEBUG_3)
        fprintf(stderr, "[DEBUG] MCT solution: makespan=%f energy=%f.\n", sol->makespan, sol->energy_consumption);
    #endif
}
