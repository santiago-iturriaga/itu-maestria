#include "mct.h"

#include "../config.h"

void compute_mct(struct solution *sol) {
    #if defined(DEBUG_0)
        fprintf(stderr, "[DEBUG] calculando MCT...\n");
    #endif
    
    sol->makespan = 0.0;

    int tasks_count = sol->etc->tasks_count;
    int machines_count = sol->etc->machines_count;
    
    float *machine_compute_time = sol->machine_compute_time;
    int *task_assignment = sol->task_assignment;

    for (int task = 0; task < tasks_count; task++) {
        int best_machine;
        best_machine = 0;

        float best_etc_value;
        best_etc_value = machine_compute_time[best_machine] 
            + get_etc_value(sol->etc, best_machine, task);

        for (int machine = 1; machine < machines_count; machine++) {
            float etc_value;
            etc_value = machine_compute_time[machine] 
                + get_etc_value(sol->etc, machine, task);

            if (etc_value < best_etc_value) {
                best_etc_value = etc_value;
                best_machine = machine;
            }
        }

        task_assignment[task] = best_machine;
        machine_compute_time[best_machine] = best_etc_value;
    }
    
    refresh_solution(sol);

    #if defined(DEBUG_0)
        fprintf(stderr, "[DEBUG] MCT solution: makespan=%f energy=%f.\n", sol->makespan, sol->energy_consumption);
    #endif
}
