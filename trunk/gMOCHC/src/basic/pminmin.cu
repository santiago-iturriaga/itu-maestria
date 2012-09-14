#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

#include "pminmin.h"

#include "../config.h"
#include "../utils.h"
#include "../global.h"

struct thread_data {
    int t_i;
    int t_f;
    struct solution *sol;
};

void* compute_pminmin_thread(void *);

void compute_pminmin(struct solution *sol) {
    #if defined(DEBUG_0)
        fprintf(stderr, "[DEBUG] calculando pMinMin/D...\n");
    #endif
    
    pthread_t threads[INPUT.thread_count];
    int rc;

    struct thread_data threadDataArray[INPUT.thread_count];

    // Create the thread pool
    int chunk = INPUT.tasks_count/INPUT.thread_count;
    for (int i = 0; i < INPUT.thread_count; i ++) {
        threadDataArray[i].t_i = i*chunk;
        threadDataArray[i].t_f = (i+1)*chunk;
        threadDataArray[i].sol = sol;
        
        rc = pthread_create(&threads[i], NULL, &compute_pminmin_thread, &threadDataArray[i]);
        if (rc) {
            fprintf(stderr, "[ERROR] return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    //wait for completion
    void* status;
    for (int k = 0; k < INPUT.thread_count; k ++) {
        rc = pthread_join(threads[k], &status);
        if (rc) {
            fprintf(stderr, "[ERROR] return code from pthread_join() is %d\n", rc);
            exit(-1);
        }
    }

    // Calcula el makespan de la solución.
    int m;
    for (int i = 0; i < INPUT.tasks_count; i++) {
        m = sol->task_assignment[i];
        sol->machine_compute_time[m] += get_etc_value(m, i);
    }

    sol->makespan = sol->machine_compute_time[0];
    for (int i = 1; i < INPUT.machines_count; i++) {
        if (sol->makespan < sol->machine_compute_time[i]) {
            sol->makespan = sol->machine_compute_time[i];
        }
    }

    refresh_solution(sol);
    sol->initialized = 1;

    #ifdef DEBUG_0
        fprintf(stderr, "[DEBUG] pMinMin/D %d-threads: makespan=%f energy=%f.\n", INPUT.thread_count, sol->makespan, sol->energy_consumption);
    #endif
}

void* compute_pminmin_thread(void *data) {
    struct thread_data *d = (struct thread_data*)data;

    int t_i = (int)d->t_i;
    int t_f = (int)d->t_f;
    struct solution *sol = d->sol;
    
    int nt = t_f - t_i;

    int assigned_tasks[nt];
    for (int i = t_i; i < t_f; i++) {
        assigned_tasks[i-t_i] = 0;
    }

    int assigned_tasks_count = 0;
    
    FLOAT *machine_compute_time = (FLOAT*)malloc(sizeof(FLOAT) * INPUT.machines_count);
    for (int i = 0; i < INPUT.machines_count; i++) machine_compute_time[i] = 0.0;

    while (assigned_tasks_count < nt) { // Mientras quede una tarea sin asignar.
        int best_task;
        int best_machine;
        FLOAT best_cost;

        best_task = -1;
        best_machine = -1;
        best_cost = 0.0;

        // Recorro las tareas.
        for (int task_i = t_i; task_i < t_f; task_i++) {
            // Si la tarea task_i no esta asignada.
            if (assigned_tasks[task_i-t_i] == 0) {
                int best_machine_for_task;
                best_machine_for_task = 0;

                FLOAT best_machine_cost_for_task;

                best_machine_cost_for_task = machine_compute_time[0] +
                    get_etc_value(0, task_i);

                for (int machine_x = 1; machine_x < INPUT.machines_count; machine_x++) {
                    FLOAT current_cost;

                    current_cost = machine_compute_time[machine_x] +
                        get_etc_value(machine_x, task_i);

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
        assigned_tasks[best_task-t_i] = 1;

        sol->task_assignment[best_task] = best_machine;
        machine_compute_time[best_machine] = best_cost;
    }
    
    free(machine_compute_time);
    
    return EXIT_SUCCESS;
}
