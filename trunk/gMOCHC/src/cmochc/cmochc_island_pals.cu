#include <assert.h>
#include <stdio.h>

#include "cmochc_island_pals.h"

#include "../config.h"
#include "../global.h"
#include "cmochc_island.h"
#include "cmochc_island_chc.h"

#ifdef DEBUG_3
    int CHC_PALS_COUNT_EXECUTIONS[MAX_THREADS] = {0};
    int CHC_PALS_COUNT_FITNESS_IMPROV[MAX_THREADS] = {0};
    int CHC_PALS_COUNT_FITNESS_DECLINE[MAX_THREADS] = {0};
#endif

void pals_init(int thread_id) {

}

void pals_free(int thread_id) {

}

void pals_search(int thread_id, int solution_index) {
    int max_tasks = INPUT.tasks_count / PALS__MAX_TASK_SEL_DIV;
    if (max_tasks == 0) max_tasks = 1;

    #ifdef DEBUG_3
        FLOAT makespan_pre = EA_THREADS[thread_id].population[solution_index].makespan;
        FLOAT energy_pre = EA_THREADS[thread_id].population[solution_index].energy_consumption;
        
        CHC_PALS_COUNT_EXECUTIONS[thread_id]++;
    #endif

    int max_et_machine = 0;
    for (int m = 1; m < INPUT.machines_count; m++) {
        if (EA_THREADS[thread_id].population[solution_index].machine_compute_time[m] >
            EA_THREADS[thread_id].population[solution_index].machine_compute_time[max_et_machine]) {

            max_et_machine = m;
        }
    }

    int src_task, src_machine;
    for (int t = 0; t < max_tasks; t++) {
        src_task = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]) * INPUT.tasks_count;
        src_machine = EA_THREADS[thread_id].population[solution_index].task_assignment[src_task];

        FLOAT etc_mov_score = 0;
        FLOAT energy_mov_score = 0;
        FLOAT mov_score = 0;

        int dst_task, best_dst_task;
        int dst_machine;

        for (int i = 0; i < PALS__MAX_INTENTOS; i++) {
            dst_task = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]) * INPUT.tasks_count;
            while (EA_THREADS[thread_id].population[solution_index].task_assignment[dst_task] == src_machine) {
                if (dst_task + 1 == INPUT.tasks_count) dst_task = 0;
                else dst_task++;
            }
            dst_machine = EA_THREADS[thread_id].population[solution_index].task_assignment[dst_task];

            if ((dst_machine == max_et_machine) || (src_machine == max_et_machine)) {
                /* La máquina con el mayor compute time esta involucrada */

                // TODO: ........... implementar mejor!!
                etc_mov_score = get_etc_value(dst_machine, src_task) - get_etc_value(src_machine, dst_task);
                energy_mov_score = (get_energy_value(dst_machine, src_task) - get_energy_value(src_machine, dst_task));

                if ((EA_THREADS[thread_id].weight_makespan * etc_mov_score) + (EA_THREADS[thread_id].weight_energy * energy_mov_score) > mov_score) {
                    mov_score = etc_mov_score + energy_mov_score;
                    best_dst_task = dst_task;
                }
            } else {
                /* Ninguna de las dos maquinas define el compute time */
                etc_mov_score = get_etc_value(dst_machine, src_task) - get_etc_value(src_machine, dst_task);
                energy_mov_score = (get_energy_value(dst_machine, src_task) - get_energy_value(src_machine, dst_task));

                if ((EA_THREADS[thread_id].weight_makespan * etc_mov_score) + (EA_THREADS[thread_id].weight_energy * energy_mov_score) > mov_score) {
                    mov_score = etc_mov_score + energy_mov_score;
                    best_dst_task = dst_task;
                }
            }
        }

        if (mov_score > 0) {
            dst_machine = EA_THREADS[thread_id].population[solution_index].task_assignment[best_dst_task];
            
            #ifdef DEBUG_3
                fprintf(stderr, "[DEBUG] <thread=%d> PALS swap task %d in %d with task %d in %d\n", thread_id, src_task, src_machine, best_dst_task, dst_machine);
            #endif
            
            /* Hago el swap */
            EA_THREADS[thread_id].population[solution_index].task_assignment[best_dst_task] = src_machine;
            EA_THREADS[thread_id].population[solution_index].task_assignment[src_task] = dst_machine;

            /* Actualizo el compute time de las maquinas involucradas */
            EA_THREADS[thread_id].population[solution_index].machine_compute_time[dst_machine] += get_etc_value(dst_machine, src_task);
            EA_THREADS[thread_id].population[solution_index].machine_compute_time[dst_machine] -= get_etc_value(dst_machine, best_dst_task);

            EA_THREADS[thread_id].population[solution_index].machine_compute_time[src_machine] += get_etc_value(src_machine, best_dst_task);
            EA_THREADS[thread_id].population[solution_index].machine_compute_time[src_machine] -= get_etc_value(src_machine, src_task);

            /* Actualizo el makespan general del schedule */
            int recompute_energy;
            recompute_energy = 0;
            
            if ((EA_THREADS[thread_id].population[solution_index].machine_compute_time[dst_machine] > EA_THREADS[thread_id].population[solution_index].makespan) ||
                (EA_THREADS[thread_id].population[solution_index].machine_compute_time[dst_machine] >= EA_THREADS[thread_id].population[solution_index].machine_compute_time[src_machine])) {
                    
                // El makespan aumentó
                EA_THREADS[thread_id].population[solution_index].makespan = EA_THREADS[thread_id].population[solution_index].machine_compute_time[dst_machine];
                max_et_machine = dst_machine;
                
                recompute_energy = 1;
            } else if ((EA_THREADS[thread_id].population[solution_index].machine_compute_time[src_machine] > EA_THREADS[thread_id].population[solution_index].makespan) ||
                (EA_THREADS[thread_id].population[solution_index].machine_compute_time[src_machine] > EA_THREADS[thread_id].population[solution_index].machine_compute_time[dst_machine])) {

                // El makespan aumentó
                EA_THREADS[thread_id].population[solution_index].makespan = EA_THREADS[thread_id].population[solution_index].machine_compute_time[src_machine];
                max_et_machine = src_machine;
                
                recompute_energy = 1;
            } else {
                if (((dst_machine == max_et_machine) && (EA_THREADS[thread_id].population[solution_index].machine_compute_time[dst_machine] < EA_THREADS[thread_id].population[solution_index].makespan)) 
                    || ((src_machine == max_et_machine) && (EA_THREADS[thread_id].population[solution_index].machine_compute_time[src_machine] < EA_THREADS[thread_id].population[solution_index].makespan))) {
                    
                    // Se redujo el makespan
                    EA_THREADS[thread_id].population[solution_index].makespan = EA_THREADS[thread_id].population[solution_index].machine_compute_time[0];
                    max_et_machine = 0;
                    
                    for (int m = 1; m < INPUT.machines_count; m++) {
                        if (EA_THREADS[thread_id].population[solution_index].machine_compute_time[0] > EA_THREADS[thread_id].population[solution_index].makespan) {
                            EA_THREADS[thread_id].population[solution_index].makespan = EA_THREADS[thread_id].population[solution_index].machine_compute_time[0];
                            max_et_machine = m;
                        }
                    }
                    
                    recompute_energy = 1;
                }
            }

            if (recompute_energy == 1) {
                /* Recalculo todo */
                
                EA_THREADS[thread_id].population[solution_index].energy_consumption = 0.0;

                for (int m = 0; m < INPUT.machines_count; m++) {
                    EA_THREADS[thread_id].population[solution_index].machine_energy_consumption[m] =
                        EA_THREADS[thread_id].population[solution_index].makespan * get_scenario_energy_idle(m);

                    EA_THREADS[thread_id].population[solution_index].machine_energy_consumption[m] +=
                        EA_THREADS[thread_id].population[solution_index].machine_compute_time[m] *
                            (get_scenario_energy_max(m) - get_scenario_energy_idle(m));

                    EA_THREADS[thread_id].population[solution_index].energy_consumption +=
                        EA_THREADS[thread_id].population[solution_index].machine_energy_consumption[m];
                }
            } else {
                /* Solo actualizo las máquinas involucradas */
                
                EA_THREADS[thread_id].population[solution_index].machine_energy_consumption[src_machine] +=
                    get_energy_value(src_machine, best_dst_task) - get_energy_value(src_machine, src_task);

                EA_THREADS[thread_id].population[solution_index].machine_energy_consumption[src_machine] +=
                    (get_etc_value(src_machine, src_task) - get_etc_value(src_machine, best_dst_task)) *
                        get_scenario_energy_idle(src_machine);

                EA_THREADS[thread_id].population[solution_index].machine_energy_consumption[dst_machine] +=
                    get_energy_value(dst_machine, src_task) - get_energy_value(dst_machine, best_dst_task);

                EA_THREADS[thread_id].population[solution_index].machine_energy_consumption[dst_machine] +=
                    (get_etc_value(dst_machine, best_dst_task) - get_etc_value(dst_machine, src_task)) *
                        get_scenario_energy_idle(dst_machine);
            }
        }
    }
       
    if (EA_THREADS[thread_id].population[solution_index].makespan < EA_THREADS[thread_id].makespan_zenith_value) {
        EA_THREADS[thread_id].makespan_zenith_value = EA_THREADS[thread_id].population[solution_index].makespan;
    } else if (EA_THREADS[thread_id].population[solution_index].makespan > EA_THREADS[thread_id].makespan_nadir_value) {
        EA_THREADS[thread_id].makespan_nadir_value = EA_THREADS[thread_id].population[solution_index].makespan;
    }

    if (EA_THREADS[thread_id].population[solution_index].energy_consumption < EA_THREADS[thread_id].energy_zenith_value) {
        EA_THREADS[thread_id].energy_zenith_value = EA_THREADS[thread_id].population[solution_index].energy_consumption;
    } else if (EA_THREADS[thread_id].population[solution_index].energy_consumption > EA_THREADS[thread_id].energy_nadir_value) {
        EA_THREADS[thread_id].energy_nadir_value = EA_THREADS[thread_id].population[solution_index].energy_consumption;
    }
    
    refresh_solution(&EA_THREADS[thread_id].population[solution_index]);
    
    EA_THREADS[thread_id].fitness_population[solution_index] = NAN;
    FLOAT fitness_post = fitness(thread_id, solution_index);
    
    #ifdef DEBUG_3   
        FLOAT fitness_pre = fitness_zn(
            thread_id, makespan_pre, energy_pre, 
            EA_THREADS[thread_id].makespan_nadir_value, EA_THREADS[thread_id].makespan_zenith_value, 
            EA_THREADS[thread_id].energy_nadir_value, EA_THREADS[thread_id].energy_zenith_value);
                   
        fprintf(stderr, "[DEBUG] <thread=%d> PALS fitness from %.4f to %.4f (makespan %.2f->%.2f / energy %.2f->%.2f)\n", 
            thread_id, fitness_pre, fitness_post, makespan_pre, EA_THREADS[thread_id].population[solution_index].makespan,
            energy_pre, EA_THREADS[thread_id].population[solution_index].energy_consumption);
        
        if (fitness_post < fitness_pre) CHC_PALS_COUNT_FITNESS_IMPROV[thread_id]++;
        if (fitness_post > fitness_pre) CHC_PALS_COUNT_FITNESS_DECLINE[thread_id]++;
    #endif
}
