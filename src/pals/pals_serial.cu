#include <stdio.h>
#include <cuda.h>
#include <math.h>

#include "../config.h"
#include "pals_serial.h"

void pals_serial_wrapper(struct matrix *etc_matrix, struct solution *s,
    ushort &best_swap_task_a, ushort &best_swap_task_b, float &best_swap_delta) {

    best_swap_task_a = 0;
    best_swap_task_b = 0;
    best_swap_delta = 0.0;

    for (ushort task_a = 0; task_a < etc_matrix->tasks_count; task_a++) {
        for (ushort task_b = 0; task_b < task_a; task_b++) {
            ushort machine_a = s->task_assignment[task_a];
            ushort machine_b = s->task_assignment[task_b];

            float current_swap_delta = 0.0;
            current_swap_delta -= get_etc_value(etc_matrix, machine_a, task_a);
            current_swap_delta -= get_etc_value(etc_matrix, machine_b, task_b);

            current_swap_delta += get_etc_value(etc_matrix, machine_a, task_b);
            current_swap_delta += get_etc_value(etc_matrix, machine_b, task_a);

            if (best_swap_delta > current_swap_delta) {
                best_swap_delta = current_swap_delta;
                best_swap_task_a = task_a;
                best_swap_task_b = task_b;
            }
        }
    }
}

void pals_serial(struct params &input, struct matrix *etc_matrix, struct solution *current_solution) {
    ushort best_swap_task_a;
    ushort best_swap_task_b;
    float best_swap_delta;

    for (int i = 0; i < PALS_COUNT; i++) {
        pals_serial_wrapper(etc_matrix, current_solution, best_swap_task_a, best_swap_task_b, best_swap_delta);
    }

    fprintf(stdout, "[DEBUG] Best swap: task %d for task %d. Gain %f.\n", best_swap_task_a, best_swap_task_b, best_swap_delta);
}
