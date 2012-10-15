#ifndef CMOCHC_ISLANDS_CHC__H
#define CMOCHC_ISLANDS_CHC__H

#include <math.h>

#include "cmochc_island.h"

#include "../config.h"
#include "../global.h"
#include "../solution.h"
#include "../load_params.h"
#include "../scenario.h"
#include "../etc_matrix.h"
#include "../energy_matrix.h"
#include "../utils.h"
#include "../basic/mct.h"
#include "../basic/minmin.h"
#include "../random/random.h"
#include "../archivers/aga.h"

//#define CHC__MUTATE_OP 0
#define CHC__MUTATE_OP 1
#define CHC__MUTATION_PROB 0.7

//#define CHC__CROSS_OP 0
#define CHC__CORSS_OP 1
#define CHC__CROSS_PROB 0.8

#if CHC__MUTATE_OP == 0
    #define CHC__MUTATE(rand_state,seed,mutation) mutate_0(rand_state,seed,mutation);
#endif
#if CHC__MUTATE_OP == 1
    #define CHC__MUTATE(rand_state,seed,mutation) mutate_1(rand_state,seed,mutation);
#endif

#if CHC__CORSS_OP == 0
    #define CHC__CROSS(rand_state, p1, p2, c1, c2) hux_0(rand_state, p1, p2, c1, c2);
#endif
#if CHC__CORSS_OP == 1
    #define CHC__CROSS(rand_state, p1, p2, c1, c2) hux_1(rand_state, p1, p2, c1, c2);
#endif

inline int distance(struct solution *s1, struct solution *s2) {
    int distance = 0;

    int tasks_count = INPUT.tasks_count;
    int *s1_task_assignment = s1->task_assignment;
    int *s2_task_assignment = s2->task_assignment;

    for (int i = 0; i < tasks_count; i++) {
        if (s1_task_assignment[i] != s2_task_assignment[i]) {
            distance++;
        }
    }

    assert(distance >= 0);
    assert(distance <= INPUT.tasks_count);

    return distance;
}

//#define CMOCHC_LOCAL__MATING_CHANCE         256
#define CMOCHC_LOCAL__MATING_CHANCE         2

inline void hux_0(RAND_STATE &rand_state,
    struct solution *p1, struct solution *p2,
    struct solution *c1, struct solution *c2) {

    FLOAT cross_prob = CMOCHC_LOCAL__MATING_CHANCE / (FLOAT)INPUT.tasks_count;

    FLOAT random;
    int current_task_index = 0;

    while (current_task_index < INPUT.tasks_count) {
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
        while ((mask_index < mask_size) && (current_task_index < INPUT.tasks_count)) {
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

    //TODO: ARREGLAR!!!
    recompute_solution(c1);
    recompute_solution(c2);
}

inline void hux_1(RAND_STATE &rand_state,
    struct solution *p1, struct solution *p2,
    struct solution *c1, struct solution *c2) {

    int tasks_count = INPUT.tasks_count;
    int machines_count = INPUT.machines_count;
    
    assert(c1 != p1);
    assert(c1 != p2);
    assert(c2 != p1);
    assert(c2 != p1);
    assert(c1 != c2);
    
    clone_solution(c1, p1);
    clone_solution(c2, p2);
    
    int *p1_task_assignment = p1->task_assignment;
    int *p2_task_assignment = p2->task_assignment;
    
    int *c1_task_assignment = c1->task_assignment;
    int *c2_task_assignment = c2->task_assignment;

    FLOAT *c1_machine_compute_time = c1->machine_compute_time;
    FLOAT *c2_machine_compute_time = c2->machine_compute_time;

    FLOAT *c1_machine_active_energy_consumption = c1->machine_active_energy_consumption;
    FLOAT *c2_machine_active_energy_consumption = c2->machine_active_energy_consumption;
    
    int cant_crossovers = machines_count;

    int task_src;
    int machine_src, machine_dst;

    for (int i = 0; i < cant_crossovers; i++) {
        task_src = (int)(RAND_GENERATE(rand_state) * tasks_count);
        
        if (c1_task_assignment[task_src] != p2_task_assignment[task_src]) {   
            machine_src = c1_task_assignment[task_src];
            machine_dst = p2_task_assignment[task_src];
            
            c1_machine_compute_time[machine_src] -= get_etc_value(machine_src, task_src);
            c1_machine_compute_time[machine_dst] += get_etc_value(machine_dst, task_src);

            c1_machine_active_energy_consumption[machine_src] -= get_scenario_energy_max(machine_src) * get_etc_value(machine_src, task_src);
            c1_machine_active_energy_consumption[machine_dst] += get_scenario_energy_max(machine_dst) * get_etc_value(machine_dst, task_src);
            
            c1_task_assignment[task_src] = machine_dst;
        }
                
        if (c2_task_assignment[task_src] != p1_task_assignment[task_src]) {
            machine_src = c2_task_assignment[task_src];
            machine_dst = p1_task_assignment[task_src];
            
            c2_machine_compute_time[machine_src] -= get_etc_value(machine_src, task_src);
            c2_machine_compute_time[machine_dst] += get_etc_value(machine_dst, task_src);

            c2_machine_active_energy_consumption[machine_src] -= get_scenario_energy_max(machine_src) * get_etc_value(machine_src, task_src);
            c2_machine_active_energy_consumption[machine_dst] += get_scenario_energy_max(machine_dst) * get_etc_value(machine_dst, task_src);
            
            c2_task_assignment[task_src] = machine_dst;
        }
    }

    c1->initialized = SOLUTION__IN_USE;
    c2->initialized = SOLUTION__IN_USE;

    recompute_metrics(c1);
    recompute_metrics(c2);
}

//#define CMOCHC_LOCAL__MUTATE_CHANCE         256 
#define CMOCHC_LOCAL__MUTATE_CHANCE         4

inline void mutate_0(RAND_STATE &rand_state, struct solution *seed, struct solution *mutation) {
    int current_task_index = 0;
    int tasks_count = INPUT.tasks_count;
    int machines_count = INPUT.machines_count;

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

    mutation->initialized = SOLUTION__IN_USE;
    //TODO: ARREGLAR!!!
    recompute_solution(mutation);
}

inline void mutate_1(RAND_STATE &rand_state, struct solution *seed_solution, struct solution *mutated_solution) {
    int tasks_count = INPUT.tasks_count;
    int machines_count = INPUT.machines_count;
    
    if (mutated_solution != seed_solution) clone_solution(mutated_solution, seed_solution);
    
    int *task_assignment = mutated_solution->task_assignment;
    FLOAT *machine_compute_time = mutated_solution->machine_compute_time;
    FLOAT *machine_active_energy_consumption = mutated_solution->machine_active_energy_consumption;
    
    int cant_mutations = machines_count;

    int task_src;
    int machine_src, machine_dst;

    for (int i = 0; i < cant_mutations; i++) {
        task_src = (int)(RAND_GENERATE(rand_state) * tasks_count);
        machine_src = task_assignment[task_src];
        machine_dst = (int)(RAND_GENERATE(rand_state) * machines_count);
        
        machine_compute_time[machine_src] -= get_etc_value(machine_src, task_src);
        machine_compute_time[machine_dst] += get_etc_value(machine_dst, task_src);

        machine_active_energy_consumption[machine_src] -= get_scenario_energy_max(machine_src) * get_etc_value(machine_src, task_src);
        machine_active_energy_consumption[machine_dst] += get_scenario_energy_max(machine_dst) * get_etc_value(machine_dst, task_src);
        
        task_assignment[task_src] = machine_dst;
    }

    mutated_solution->initialized = SOLUTION__IN_USE;
    recompute_metrics(mutated_solution);
}

extern inline FLOAT fitness_zn(int thread_id, 
    FLOAT makespan_value, FLOAT energy_value,
    FLOAT makespan_nadir_value, FLOAT makespan_zenith_value, 
    FLOAT energy_nadir_value, FLOAT energy_zenith_value) {

    FLOAT fitness;

    if (makespan_nadir_value > makespan_zenith_value) {
        fitness = ((makespan_value - makespan_zenith_value) / (makespan_nadir_value - makespan_zenith_value)) * 
            EA_THREADS[thread_id].weight_makespan;
    } else {
        fitness = (makespan_value - makespan_zenith_value) * EA_THREADS[thread_id].weight_makespan;
    }
            
    if (energy_nadir_value > energy_zenith_value) {
        fitness += ((energy_value - energy_zenith_value) / (energy_nadir_value - energy_zenith_value)) * 
            EA_THREADS[thread_id].weight_energy;
    } else {
        fitness += (energy_value - energy_zenith_value) * EA_THREADS[thread_id].weight_energy;
    }
    
    return fitness;
}

inline FLOAT fitness(int thread_id, int solution_index) {
    #ifdef CMOCHC_LOCAL__Z_FITNESS_NORM
        if (isnan(EA_THREADS[thread_id].fitness_population[solution_index])) {
            EA_THREADS[thread_id].fitness_population[solution_index] =
                ((EA_THREADS[thread_id].population[solution_index].makespan /
                    EA_THREADS[thread_id].makespan_zenith_value) * EA_THREADS[thread_id].weight_makespan) +
                ((EA_THREADS[thread_id].population[solution_index].energy_consumption /
                    EA_THREADS[thread_id].energy_zenith_value) * EA_THREADS[thread_id].energy_makespan);
        }
    #endif
    #ifdef CMOCHC_LOCAL__ZN_FITNESS_NORM
        if (isnan(EA_THREADS[thread_id].fitness_population[solution_index])) {           
            if (EA_THREADS[thread_id].makespan_nadir_value > EA_THREADS[thread_id].makespan_zenith_value) {
                EA_THREADS[thread_id].fitness_population[solution_index] =
                    (((EA_THREADS[thread_id].population[solution_index].makespan - EA_THREADS[thread_id].makespan_zenith_value) /
                        (EA_THREADS[thread_id].makespan_nadir_value - EA_THREADS[thread_id].makespan_zenith_value)) * EA_THREADS[thread_id].weight_makespan);
            } else {
                EA_THREADS[thread_id].fitness_population[solution_index] =
                    ((EA_THREADS[thread_id].population[solution_index].makespan - EA_THREADS[thread_id].makespan_zenith_value) * 
                        EA_THREADS[thread_id].weight_makespan);
            }
                    
            if (EA_THREADS[thread_id].energy_nadir_value > EA_THREADS[thread_id].energy_zenith_value) {
                EA_THREADS[thread_id].fitness_population[solution_index] +=
                    (((EA_THREADS[thread_id].population[solution_index].energy_consumption - EA_THREADS[thread_id].energy_zenith_value) /
                        (EA_THREADS[thread_id].energy_nadir_value - EA_THREADS[thread_id].energy_zenith_value)) * EA_THREADS[thread_id].weight_energy);
            } else {
                EA_THREADS[thread_id].fitness_population[solution_index] +=
                    ((EA_THREADS[thread_id].population[solution_index].energy_consumption - EA_THREADS[thread_id].energy_zenith_value) * 
                        EA_THREADS[thread_id].weight_energy);
            }
        }
    #endif

    return EA_THREADS[thread_id].fitness_population[solution_index];
}

inline void fitness_all(int thread_id) {
    for (int i = 0; i < MAX_POP_SOLS; i++) {
        if (isnan(EA_THREADS[thread_id].fitness_population[i])) {
            fitness(thread_id, i);
        }
    }
}

inline void fitness_reset(int thread_id) {
    for (int i = 0; i < MAX_POP_SOLS; i++) {
        EA_THREADS[thread_id].fitness_population[i] = NAN;
    }
}

void chc_population_init(int thread_id);
void chc_evolution(int thread_id);

#endif // CMOCHC_ISLANDS_CHC__H
