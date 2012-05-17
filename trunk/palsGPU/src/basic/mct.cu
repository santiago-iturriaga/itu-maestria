#include "../config.h"
#include "mct.h"

void compute_mct(struct matrix *etc_matrix, struct solution *solution) {
    solution->makespan = 0.0;

    int tasks_count = etc_matrix->tasks_count;
    int machines_count = etc_matrix->machines_count;
    float *machine_compute_time = solution->machine_compute_time;
    int *task_assignment = solution->task_assignment;

    for (int task = 0; task < tasks_count; task++) {
        int best_machine;
        best_machine = 0;

        float best_etc_value;
        best_etc_value = machine_compute_time[best_machine] 
            + get_etc_value(etc_matrix, best_machine, task);

        for (int machine = 1; machine < machines_count; machine++) {
            float etc_value;
            etc_value = machine_compute_time[machine] + get_etc_value(etc_matrix, machine, task);

            if (etc_value < best_etc_value) {
                best_etc_value = etc_value;
                best_machine = machine;
            }
        }

        task_assignment[task] = best_machine;
        machine_compute_time[best_machine] = best_etc_value;

        if (machine_compute_time[best_machine] > solution->makespan) {
            solution->makespan = machine_compute_time[best_machine];
        }
    }

    #if defined(DEBUG)
        fprintf(stdout, "[DEBUG] MCT Solution makespan: %f.\n", solution->makespan);
    #endif
}
