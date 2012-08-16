#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "solution.h"
#include "config.h"

void create_empty_solution(struct solution *new_solution, struct scenario *s, struct etc_matrix *etc, struct energy_matrix *energy) {   
    new_solution->initialized = 0;
    new_solution->s = s;
    new_solution->etc = etc;
    new_solution->energy = energy;

    new_solution->task_assignment = (int*)(malloc(sizeof(int) * etc->tasks_count));

    if (new_solution->task_assignment == NULL) {
        fprintf(stderr, "[ERROR] solicitando memoria para el new_solution->task_assignment.\n");
        exit(EXIT_FAILURE);
    }

    for (int task = 0; task < etc->tasks_count; task++) {
        new_solution->task_assignment[task] = SOLUTION__TASK_NOT_ASSIGNED; /* not yet assigned */
    }

    //=== Estructura de machine compute time.
    new_solution->makespan = 0.0;
    new_solution->machine_compute_time = (float*)(malloc(sizeof(float) * etc->machines_count));

    if (new_solution->machine_compute_time == NULL) {
        fprintf(stderr, "[ERROR] solicitando memoria para el new_solution->__machine_compute_time.\n");
        exit(EXIT_FAILURE);
    }

    for (int machine = 0; machine < etc->machines_count; machine++) {
        new_solution->machine_compute_time[machine] = 0.0;
    }

    //=== Estructura de energy.
    new_solution->energy_consumption = 0.0;
    new_solution->machine_energy_consumption = (float*)(malloc(sizeof(float) * etc->machines_count));

    if (new_solution->machine_energy_consumption == NULL) {
        fprintf(stderr, "[ERROR] solicitando memoria para el new_solution->__machine_energy_consumption.\n");
        exit(EXIT_FAILURE);
    }

    for (int machine = 0; machine < etc->machines_count; machine++) {
        new_solution->machine_energy_consumption[machine] = 0.0;
    }
}

void clone_solution(struct solution *dst, struct solution *src) {
    dst->s = src->s;
    dst->etc = src->etc;
    dst->energy = src->energy;
    
    dst->makespan = src->makespan;
    dst->energy_consumption = src->energy_consumption;
    
    memcpy(dst->task_assignment, src->task_assignment, sizeof(int) * src->etc->tasks_count);
    memcpy(dst->machine_compute_time, src->machine_compute_time, sizeof(float) * src->etc->machines_count);
    memcpy(dst->machine_energy_consumption, src->machine_energy_consumption, sizeof(float) * src->etc->machines_count);
}

void free_solution(struct solution *sol) {
    free(sol->task_assignment);
    free(sol->machine_compute_time);
    free(sol->machine_energy_consumption);
}

void refresh_solution(struct solution *sol) {
    for (int machine = 0; machine < sol->etc->machines_count; machine++) {
        sol->machine_compute_time[machine] = 0.0;
    }
    
    sol->makespan = 0.0;
    
    for (int task = 0; task < sol->etc->tasks_count; task++) {
        int machine = sol->task_assignment[task];
        
        sol->machine_compute_time[machine] += get_etc_value(sol->etc, machine, task);
        if (sol->makespan < sol->machine_compute_time[machine]) {
            sol->makespan = sol->machine_compute_time[machine];
        }
    }

    for (int machine = 0; machine < sol->etc->machines_count; machine++) {
        sol->machine_energy_consumption[machine] = sol->makespan * get_scenario_energy_idle(sol->s, machine);
        sol->energy_consumption += sol->machine_energy_consumption[machine];
    }
    
    double task_energy_consumption;
    for (int task = 0; task < sol->etc->tasks_count; task++) {
        int machine = sol->task_assignment[task];
        
        task_energy_consumption = get_energy_value(sol->energy, machine, task);
        
        sol->machine_energy_consumption[machine] += task_energy_consumption;
        sol->energy_consumption += task_energy_consumption;
    }
}

void validate_solution(struct solution *s) {
    fprintf(stderr, "[INFO] Validate solution =========================== \n");
    fprintf(stderr, "[INFO] Dimension (%d x %d).\n", s->etc->tasks_count, s->etc->machines_count);
    fprintf(stderr, "[INFO] Makespan: %f.\n", s->makespan);

    {
        float aux_makespan = 0.0;

        for (int machine = 0; machine < s->etc->machines_count; machine++) {
            if (aux_makespan < s->machine_compute_time[machine]) {
                aux_makespan = s->machine_compute_time[machine];
            }
        }

        assert(s->makespan == aux_makespan);
    }

    for (int task = 0; task < s->etc->tasks_count; task++) {
        assert(s->task_assignment[task] >= 0);
        assert(s->task_assignment[task] < s->etc->machines_count);
    }

    fprintf(stderr, "[INFO] The current solution is valid.\n");
    fprintf(stderr, "[INFO] ============================================= \n");
}

void show_solution(struct solution *s) {
    fprintf(stderr, "[INFO] Show solution =========================== \n");
    fprintf(stderr, "[INFO] Dimension (%d x %d).\n", s->etc->tasks_count, s->etc->machines_count);

    fprintf(stderr, "   Makespan: %f.\n", s->makespan);

    for (int machine = 0; machine < s->etc->machines_count; machine++) {
        fprintf(stderr, "   Machine: %d -> execution time: %f.\n", machine, s->machine_compute_time[machine]);
    }

    for (int task = 0; task < s->etc->tasks_count; task++) {
        fprintf(stderr, "   Task: %d -> assigned to: %d.\n", task, s->task_assignment[task]);
    }
    fprintf(stderr, "[INFO] ========================================= \n");
}
