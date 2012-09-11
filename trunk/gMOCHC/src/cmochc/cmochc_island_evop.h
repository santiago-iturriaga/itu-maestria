#if !defined(CMOCHC_ISLANDS_EVOP__H)
#define CMOCHC_ISLANDS_EVOP__H

#include <math.h>

#include "../config.h"
#include "../solution.h"
#include "../load_params.h"
#include "../scenario.h"
#include "../etc_matrix.h"
#include "../energy_matrix.h"
#include "../utils.h"
#include "../basic/mct.h"
#include "../random/random.h"
#include "../archivers/aga.h"

#include "cmochc_island.h"

inline int distance(struct solution *s1, struct solution *s2) {
    int distance = 0;

    for (int i = 0; i < s1->etc->tasks_count; i++) {
        if (s1->task_assignment[i] != s2->task_assignment[i]) {
            distance++;
        }
    }

    assert(distance >= 0);
    assert(distance <= s1->etc->tasks_count);

    return distance;
}

inline void hux(RAND_STATE &rand_state,
    struct solution *p1, struct solution *p2,
    struct solution *c1, struct solution *c2) {

    FLOAT cross_prob = CMOCHC_LOCAL__MATING_CHANCE / (FLOAT)p1->etc->tasks_count;

    FLOAT random;
    int current_task_index = 0;

    while (current_task_index < p1->etc->tasks_count) {
        random = RAND_GENERATE(rand_state);

        int mask = 0x0;
        int mask_size = 256; // 8-bit mask
        FLOAT base_step = 1.0/(FLOAT)mask_size;
        FLOAT base = base_step;

        while (random > base) {
            base += base_step;
            mask += 0x1;
        }

        int mask_index = 0x1;
        while ((mask_index < mask_size) && (current_task_index < p1->etc->tasks_count)) {
            if ((mask & 0x1) == 1) {
                random = RAND_GENERATE(rand_state);

                if (random < cross_prob) {
                    // Si la máscara vale 1 copio las asignaciones cruzadas de la tarea
                    c1->task_assignment[current_task_index] = p2->task_assignment[current_task_index];
                    c2->task_assignment[current_task_index] = p1->task_assignment[current_task_index];
                } else {
                    // Si la máscara vale 0 copio las asignaciones derecho de la tarea
                    c1->task_assignment[current_task_index] = p1->task_assignment[current_task_index];
                    c2->task_assignment[current_task_index] = p2->task_assignment[current_task_index];
                }
            } else {
                // Si la máscara vale 0 copio las asignaciones derecho de la tarea
                c1->task_assignment[current_task_index] = p1->task_assignment[current_task_index];
                c2->task_assignment[current_task_index] = p2->task_assignment[current_task_index];
            }

            // Desplazo la máscara hacia la derecha
            mask = mask >> 1;
            mask_index = mask_index << 1;
            current_task_index++;
        }
    }

    c1->initialized = SOLUTION__IN_USE;
    c2->initialized = SOLUTION__IN_USE;

    refresh_solution(c1);
    refresh_solution(c2);
}

inline void mutate(RAND_STATE &rand_state, struct solution *seed, struct solution *mutation) {
    int current_task_index = 0;
    int tasks_count = seed->etc->tasks_count;
    int machines_count = seed->etc->machines_count;

    FLOAT mut_prob = CMOCHC_LOCAL__MUTATE_CHANCE / (FLOAT)tasks_count;

    while (current_task_index < tasks_count) {
        FLOAT random;
        random = RAND_GENERATE(rand_state);

        int mask = 0x0;
        int mask_size = 256; // 8-bit mask
        FLOAT base_step = 1.0/(FLOAT)mask_size;
        FLOAT base = base_step;

        while (random > base) {
            base += base_step;
            mask += 0x1;
        }

        int destination_machine;
        int mask_index = 0x1;
        while ((mask_index < mask_size) && (current_task_index < tasks_count)) {
            if ((mask & 0x1) == 1) {
                random = RAND_GENERATE(rand_state);

                if (random < mut_prob) {
                    random = RAND_GENERATE(rand_state);
                    destination_machine = (int)(floor(machines_count * random));

                    assert(destination_machine >= 0);
                    assert(destination_machine < machines_count);

                    // Si la máscara vale 1 copio reubico aleariamente la tarea
                    mutation->task_assignment[current_task_index] = destination_machine;
                } else {
                    // Si la máscara vale 0 copio las asignaciones derecho de la tarea
                    mutation->task_assignment[current_task_index] = seed->task_assignment[current_task_index];
                }
            } else {
                // Si la máscara vale 0 copio las asignaciones derecho de la tarea
                mutation->task_assignment[current_task_index] = seed->task_assignment[current_task_index];
            }

            // Desplazo la máscara hacia la derecha
            mask = mask >> 1;
            mask_index = mask_index << 1;
            current_task_index++;
        }
    }

    refresh_solution(mutation);
}

inline FLOAT fitness(struct solution *population, FLOAT *fitness_population, 
    FLOAT weight_makespan, FLOAT energy_makespan,
    FLOAT makespan_zenith_value, FLOAT energy_zenith_value,
    FLOAT makespan_nadir_value, FLOAT energy_nadir_value,
    int solution_index) {

    #ifdef CMOCHC_LOCAL__Z_FITNESS_NORM
        if (isnan(fitness_population[solution_index])) {
            fitness_population[solution_index] =
                ((population[solution_index].makespan/makespan_zenith_value) * weight_makespan) +
                ((population[solution_index].energy_consumption/energy_zenith_value) * energy_makespan);
        }
    #endif
    #ifdef CMOCHC_LOCAL__ZN_FITNESS_NORM
        if (isnan(fitness_population[solution_index])) {
            fitness_population[solution_index] =
                (((population[solution_index].makespan - makespan_zenith_value) /
                    (makespan_nadir_value - makespan_zenith_value)) * weight_makespan) +
                (((population[solution_index].energy_consumption - energy_zenith_value) /
                    (energy_nadir_value - energy_zenith_value)) * energy_makespan);
        }
    #endif

    return fitness_population[solution_index];
}

#endif // CMOCHC_ISLANDS_EVOP__H
