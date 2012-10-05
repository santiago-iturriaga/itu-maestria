#include <assert.h>
#include <stdio.h>

#include "cmochc_island_pals.h"

#include "../config.h"
#include "../global.h"
#include "cmochc_island.h"
#include "cmochc_island_chc.h"

#ifdef DEBUG_1
    int CHC_PALS_COUNT_EXECUTIONS[MAX_THREADS] = {0};
    int CHC_PALS_COUNT_FITNESS_IMPROV[MAX_THREADS] = {0};
    int CHC_PALS_COUNT_FITNESS_DECLINE[MAX_THREADS] = {0};
#endif

#define PALS_MOVIMIENTO_SWAP 0
#define PALS_MOVIMIENTO_MOVE 1

void pals_init(int thread_id) {

}

void pals_free(int thread_id) {

}

void pals_search(int thread_id, int solution_index) {
    #ifdef DEBUG_3
        fprintf(stderr, "[INFO] =======> PALS!\n");
    #endif
    #ifdef DEBUG_1
        FLOAT makespan_pre = EA_THREADS[thread_id].population[solution_index].makespan;
        FLOAT energy_pre = EA_THREADS[thread_id].population[solution_index].energy_consumption;

        CHC_PALS_COUNT_EXECUTIONS[thread_id]++;
    #endif

    // Busco la máquina con mayor compute time.
    int max_et_machine = 0;
    for (int m = 1; m < INPUT.machines_count; m++) {
        if (EA_THREADS[thread_id].population[solution_index].machine_compute_time[m] >
            EA_THREADS[thread_id].population[solution_index].machine_compute_time[max_et_machine]) {

            max_et_machine = m;
            
        } else if (EA_THREADS[thread_id].population[solution_index].machine_compute_time[m] ==
            EA_THREADS[thread_id].population[solution_index].machine_compute_time[max_et_machine]) {
                
            if (RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]) > 0.5) {
                max_et_machine = m;
            }
        }
    }

    FLOAT delta;
    int movimiento;

    FLOAT best_delta = VERY_BIG_FLOAT;
    int best_movimiento = -1;
    int best_src_task = -1;
    int best_dst = -1;

    FLOAT random1, random2;
    FLOAT current_makespan = EA_THREADS[thread_id].population[solution_index].makespan;

    for (int loop = 0; loop < PALS__MAX_INTENTOS; loop++) {
        movimiento = PALS_MOVIMIENTO_SWAP;
        //movimiento = (int)(RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]) * 2);

        if (movimiento == PALS_MOVIMIENTO_SWAP) {
            // =================
            // Swap
            int task_x, task_y;
            int machine_a, machine_b;

            FLOAT machine_a_ct_old, machine_b_ct_old;
            FLOAT machine_a_ct_new, machine_b_ct_new;

            delta = 0.0;

            // Obtengo las tareas sorteadas.
            random1 = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]);
            task_x = (int)(random1 * INPUT.tasks_count);
            machine_a = EA_THREADS[thread_id].population[solution_index].task_assignment[task_x]; // Máquina a.
            
            for (int i = 0; (i < INPUT.tasks_count) && (machine_a != max_et_machine); i++) {
                task_x++;
                if (task_x == INPUT.tasks_count) task_x = 0;
                
                machine_a = EA_THREADS[thread_id].population[solution_index].task_assignment[task_x]; // Máquina a.
            }
        
            random2 = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]);    
            task_y = (int)(random2 * INPUT.tasks_count);
            machine_b = EA_THREADS[thread_id].population[solution_index].task_assignment[task_y]; // Máquina b.
            
            for (int i = 0; (i < INPUT.tasks_count) && (machine_a == machine_b); i++) {
                task_y++;
                if (task_y == INPUT.tasks_count) task_y = 0;
                
                machine_b = EA_THREADS[thread_id].population[solution_index].task_assignment[task_y]; // Máquina b.
            }

            // Calculo el delta del swap sorteado.

            // Máquina 1.
            machine_a_ct_old = EA_THREADS[thread_id].population[solution_index].machine_compute_time[machine_a];

            machine_a_ct_new = machine_a_ct_old;
            machine_a_ct_new -= get_etc_value(machine_a, task_x); // Resto del ETC de x en a.
            machine_a_ct_new += get_etc_value(machine_a, task_y); // Sumo el ETC de y en a.

            // Máquina 2.
            machine_b_ct_old = EA_THREADS[thread_id].population[solution_index].machine_compute_time[machine_b];

            machine_b_ct_new = machine_b_ct_old;
            machine_b_ct_new -= get_etc_value(machine_b, task_y); // Resto el ETC de y en b.
            machine_b_ct_new += get_etc_value(machine_b, task_x); // Sumo el ETC de x en b.

            #ifdef DEBUG_3
                fprintf(stderr, "[DEBUG] Swap %d with %d (makespan=%.2f)\n", 
                    task_x, task_y, EA_THREADS[thread_id].population[solution_index].makespan);
                fprintf(stderr, "[DEBUG] Machine CT (a) %.2f to %.2f (b) %.2f to %.2f\n", 
                    machine_a_ct_old, machine_a_ct_new,
                    machine_b_ct_old, machine_b_ct_new);
            #endif

            FLOAT max_old;
            max_old = machine_a_ct_old;
            if (max_old < machine_b_ct_old) max_old = machine_b_ct_old;

            #if defined(SIMPLE_DELTA)
                if ((machine_a_ct_new > max_old) || (machine_b_ct_new > max_old)) {
                    delta = VERY_BIG_FLOAT - (max_old - machine_a_ct_new) + (max_old - machine_b_ct_new);
                } else {
                    delta = (machine_a_ct_new - max_old) + (machine_b_ct_new - max_old);
                }
            #endif
            #if defined(COMPLEX_DELTA)
                if ((machine_a_ct_new > current_makespan) || (machine_b_ct_new > current_makespan)) {
                    // Luego del movimiento aumenta el makespan. Intento desestimularlo lo más posible.
                    if (machine_a_ct_new > current_makespan) delta = delta + (machine_a_ct_new - current_makespan);
                    if (machine_b_ct_new > current_makespan) delta = delta + (machine_b_ct_new - current_makespan);
                } else if ((machine_a_ct_old+1 >= current_makespan) || (machine_b_ct_old+1 >= current_makespan)) {
                    // Antes del movimiento una las de máquinas definía el makespan. Estos son los mejores movimientos.
                    if (machine_a_ct_old+1 >= current_makespan) {
                        delta = delta + (machine_a_ct_new - machine_a_ct_old);
                    } else {
                        delta = delta + 1 / (machine_a_ct_new - machine_a_ct_old);
                    }

                    if (machine_b_ct_old+1 >= current_makespan) {
                        delta = delta + (machine_b_ct_new - machine_b_ct_old);
                    } else {
                        delta = delta + 1 / (machine_b_ct_new - machine_b_ct_old);
                    }
                } else {
                    // Ninguna de las máquinas intervenía en el makespan. Intento favorecer lo otros movimientos.
                    delta = delta + (machine_a_ct_new - machine_a_ct_old);
                    delta = delta + (machine_b_ct_new - machine_b_ct_old);
                    delta = 1 / delta;
                }
                
                #ifdef DEBUG_3
                    fprintf(stderr, "[DEBUG] Delta %.2f\n", delta);
                #endif
            #endif

            if (best_delta > delta) {
                best_delta = delta;
                best_movimiento = PALS_MOVIMIENTO_SWAP;
                best_src_task = task_x;
                best_dst = task_y;
            }
        } else if (movimiento == PALS_MOVIMIENTO_MOVE) {
            // =================
            // Move

            int task_x;
            int machine_a, machine_b;

            float machine_a_ct_old, machine_b_ct_old;
            float machine_a_ct_new, machine_b_ct_new;

            delta = 0.0;

            // ================= Obtengo la tarea sorteada, la máquina a la que esta asignada,
            // ================= y el compute time de la máquina.
            task_x = (int)(random1 * INPUT.tasks_count);
            machine_a = EA_THREADS[thread_id].population[solution_index].task_assignment[task_x]; // Máquina a.

            // ================= Obtengo la máquina destino sorteada.
            machine_b = (int)(random2 * INPUT.machines_count);

            if (machine_a != machine_b) {
                machine_a_ct_old = EA_THREADS[thread_id].population[solution_index].machine_compute_time[machine_a];
                machine_b_ct_old = EA_THREADS[thread_id].population[solution_index].machine_compute_time[machine_b];

                // Calculo el delta del swap sorteado.
                machine_a_ct_new = machine_a_ct_old - get_etc_value(machine_a, task_x); // Resto del ETC de x en a.
                machine_b_ct_new = machine_b_ct_old + get_etc_value(machine_b, task_x); // Sumo el ETC de x en b.

                #if defined(SIMPLE_DELTA)
                    float max_old;
                    max_old = machine_a_ct_old;
                    if (max_old < machine_b_ct_old) max_old = machine_b_ct_old;

                    if ((machine_a_ct_new > max_old) || (machine_b_ct_new > max_old)) {
                        delta = VERY_BIG_FLOAT - (max_old - machine_a_ct_new) + (max_old - machine_b_ct_new);
                    } else {
                        delta = (machine_a_ct_new - max_old) + (machine_b_ct_new - max_old);
                    }
                #endif
                #if defined(COMPLEX_DELTA)
                    if (machine_b_ct_new > current_makespan) {
                        // Luego del movimiento aumenta el makespan. Intento desestimularlo lo más posible.
                        delta = delta + (machine_b_ct_new - current_makespan);
                    } else if (machine_a_ct_old+1 >= current_makespan) {
                        // Antes del movimiento una las de máquinas definía el makespan. Estos son los mejores movimientos.
                        delta = delta + (machine_a_ct_new - machine_a_ct_old);
                        delta = delta + 1 / (machine_b_ct_new - machine_b_ct_old);
                    } else {
                        // Ninguna de las máquinas intervenía en el makespan. Intento favorecer lo otros movimientos.
                        delta = delta + (machine_a_ct_new - machine_a_ct_old);
                        delta = delta + (machine_b_ct_new - machine_b_ct_old);
                        delta = 1 / delta;
                    }
                #endif
            }

            if (best_delta > delta) {
                best_delta = delta;
                best_movimiento = PALS_MOVIMIENTO_MOVE;
                best_src_task = task_x;
                best_dst = machine_b;
            }
        }
    }

    assert(best_movimiento >= 0);
    assert(best_src_task >= 0);
    assert(best_dst >= 0);

    #ifdef DEBUG_3
        fprintf(stderr, "[DEBUG] best_delta = %.2f\n", best_delta);
    #endif

    if (best_delta <= 1) {
        int src_machine;
        src_machine = EA_THREADS[thread_id].population[solution_index].task_assignment[best_src_task];
        
        int dst_machine;

        if (best_movimiento == PALS_MOVIMIENTO_SWAP) {
            dst_machine = EA_THREADS[thread_id].population[solution_index].task_assignment[best_dst];

            /* Hago el swap */
            EA_THREADS[thread_id].population[solution_index].task_assignment[best_dst] = src_machine;
            EA_THREADS[thread_id].population[solution_index].task_assignment[best_src_task] = dst_machine;

        } else if (best_movimiento == PALS_MOVIMIENTO_MOVE) {
            dst_machine = best_dst;
            
            /* Hago el move */
            EA_THREADS[thread_id].population[solution_index].task_assignment[best_src_task] = dst_machine;

        } else {
            assert(1 == 0);
            dst_machine = -1;
        }
    }

    refresh_solution(&EA_THREADS[thread_id].population[solution_index]);

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

    EA_THREADS[thread_id].fitness_population[solution_index] = NAN;

    #ifdef DEBUG_1
        FLOAT fitness_post = fitness(thread_id, solution_index);
        
        FLOAT fitness_pre = fitness_zn(
            thread_id, makespan_pre, energy_pre,
            EA_THREADS[thread_id].makespan_nadir_value, EA_THREADS[thread_id].makespan_zenith_value,
            EA_THREADS[thread_id].energy_nadir_value, EA_THREADS[thread_id].energy_zenith_value);

        #ifdef DEBUG_3
            fprintf(stderr, "[DEBUG] <thread=%d> PALS fitness from %.4f to %.4f (makespan %.2f->%.2f / energy %.2f->%.2f)\n",
                thread_id, fitness_pre, fitness_post, makespan_pre, EA_THREADS[thread_id].population[solution_index].makespan,
                energy_pre, EA_THREADS[thread_id].population[solution_index].energy_consumption);
        #endif

        if (fitness_post < fitness_pre) CHC_PALS_COUNT_FITNESS_IMPROV[thread_id]++;
        else CHC_PALS_COUNT_FITNESS_DECLINE[thread_id]++;
        //if (fitness_post > fitness_pre) CHC_PALS_COUNT_FITNESS_DECLINE[thread_id]++;
    #endif
}
