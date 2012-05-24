#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>

#include "../config.h"
#include "../utils.h"

#include "../config.h"
#include "pminmin.h"

#include <unistd.h>

void compute_pminmin(struct etc_matrix *etc, struct solution *sol, int numberOfThreads) {
    // Timming -----------------------------------------------------
    timespec ts;
    timming_start(ts);
    // Timming -----------------------------------------------------

    pthread_t threads[numberOfThreads];
    int rc;

    struct threadData threadDataArray[numberOfThreads];

    // Create the thread pool
    int chunk = etc->tasks_count/numberOfThreads;
    //fprintf(stdout, "[DEBUG] Chunks %d\n",chunk);
    for (int i = 0; i < numberOfThreads; i ++) {
        threadDataArray[i].t_i = i*chunk;
        threadDataArray[i].t_f = (i+1)*chunk;
        threadDataArray[i].etc = etc;
        threadDataArray[i].sol = sol;
        
        rc = pthread_create(&threads[i], NULL, &compute_pminmin_thread, &threadDataArray[i]);
        if (rc) {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    //wait for completion
    void* status;
    for (int k = 0; k < numberOfThreads; k ++) {
        rc = pthread_join(threads[k], &status);
        if (rc) {
            printf("ERROR; return code from pthread_join() is %d\n", rc);
            exit(-1);
        }
    }

    refresh(sol);

    #ifdef DEBUG 
    fprintf(stdout, "[DEBUG] pMinMin %d-threads solution makespan: %f.\n", numberOfThreads, sol->__makespan);
    #endif
    
    // Timming -----------------------------------------------------
    timming_end("pMinMin time", ts);
    // Timming -----------------------------------------------------
}

void* compute_pminmin_thread(void *data) {
    struct threadData *d = (struct threadData *)data;

    int t_i = (int)d->t_i;
    int t_f = (int)d->t_f;
    struct etc_matrix *etc = d->etc;
    struct solution *sol = d->sol;
    
    int nt = t_f - t_i;

    int assigned_tasks[nt];
    for (int i = t_i; i < t_f; i++) {
        assigned_tasks[i-t_i] = 0;
    }

    int assigned_tasks_count = 0;
    
    float *machine_compute_time = (float*)malloc(sizeof(float) * etc->machines_count); //solution_l->machine_compute_time;
    for (int i = 0; i < etc->machines_count; i++) machine_compute_time[i] = 0.0;

    while (assigned_tasks_count < nt) { // Mientras quede una tarea sin asignar.
        int best_task;
        int best_machine;
        float best_cost;

        best_task = -1;
        best_machine = -1;
        best_cost = 0.0;

        // Recorro las tareas.
        for (int task_i = t_i; task_i < t_f; task_i++) {
            // Si la tarea task_i no esta asignada.
            if (assigned_tasks[task_i-t_i] == 0) {
                int best_machine_for_task;
                best_machine_for_task = 0;

                float best_machine_cost_for_task;

                best_machine_cost_for_task = machine_compute_time[0] +
                    get_etc_value(etc, 0, task_i);

                for (int machine_x = 1; machine_x < etc->machines_count; machine_x++) {
                    float current_cost;

                    current_cost = machine_compute_time[machine_x] +
                        get_etc_value(etc, machine_x, task_i);

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

        assign_task_to_machine(sol, best_machine, best_task);

        machine_compute_time[best_machine] = best_cost;
    }
    
    free(machine_compute_time);
    
    return EXIT_SUCCESS;
}
