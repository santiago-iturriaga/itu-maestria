#include "minmin.h"

#include "../config.h"
#include "../utils.h"

void compute_minmin(struct solution *sol) {
    #if defined(DEBUG_0)
        fprintf(stderr, "[DEBUG] calculando MinMin...\n");
    #endif

    int assigned_tasks[sol->etc->tasks_count];
    for (int i = 0; i < sol->etc->tasks_count; i++) {
        assigned_tasks[i] = 0;
    }

    int assigned_tasks_count = 0;

    while (assigned_tasks_count < sol->etc->tasks_count) { // Mientras quede una tarea sin asignar.
        int best_task;
        int best_machine;
        FLOAT best_cost;

        best_task = -1;
        best_machine = -1;
        best_cost = 0.0;

        // Recorro las tareas.
        for (int task_i = 0; task_i < sol->etc->tasks_count; task_i++) {
            // Si la tarea task_i no esta asignada.
            if (assigned_tasks[task_i] == 0) {
                int best_machine_for_task;
                best_machine_for_task = 0;

                FLOAT best_machine_cost_for_task;

                best_machine_cost_for_task = sol->machine_compute_time[0] +
                    get_etc_value(sol->etc, 0, task_i);

                for (int machine_x = 1; machine_x < sol->etc->machines_count; machine_x++) {
                    FLOAT current_cost;

                    current_cost = sol->machine_compute_time[machine_x] +
                        get_etc_value(sol->etc, machine_x, task_i);

                    if (current_cost < best_machine_cost_for_task) {
                        best_machine_cost_for_task = current_cost;
                        best_machine_for_task = machine_x;
                    }
                }

                if ((best_machine_cost_for_task < best_cost) || (best_task < 0)) {
                    best_task = task_i;
                    best_machine = best_machine_for_task;
                    best_cost = best_machine_cost_for_task;
                }
            }
        }

        assigned_tasks_count++;
        assigned_tasks[best_task] = 1;

        sol->task_assignment[best_task] = best_machine;

        sol->machine_compute_time[best_machine] = sol->machine_compute_time[best_machine] +
            get_etc_value(sol->etc, best_machine, best_task);
    }

    refresh_solution(sol);
    sol->initialized = 1;

    #if defined(DEBUG_0)
        fprintf(stderr, "[DEBUG] MinMin solution: makespan=%f energy=%f.\n", sol->makespan, sol->energy_consumption);
    #endif
}
