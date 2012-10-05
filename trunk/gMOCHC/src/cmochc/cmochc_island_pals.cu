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

#define PALS_MOVIMIENTO_SWAP 0
#define PALS_MOVIMIENTO_MOVE 1

void pals_init(int thread_id) {

}

void pals_free(int thread_id) {

}

void pals_search(int thread_id, int solution_index) {
    #ifdef DEBUG_3
        fprintf(stderr, "[INFO] =======> PALS!\n");
    
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
        }
    }

    FLOAT random1 = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]);
    FLOAT random2 = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]);

    FLOAT delta;
    int movimiento;

    FLOAT best_delta = 0.0;
    int best_movimiento = -1;
    int best_src_task = -1;
    int best_dst = -1;

    FLOAT current_makespan = EA_THREADS[thread_id].population[solution_index].makespan;

    for (int loop = 0; loop < PALS__MAX_INTENTOS; loop++) {
        movimiento = (int)(RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]) * 2);

        if (movimiento == PALS_MOVIMIENTO_SWAP) {
            // =================
            // Swap

            int task_x, task_y;
            int machine_a, machine_b;

            FLOAT machine_a_ct_old, machine_b_ct_old;
            FLOAT machine_a_ct_new, machine_b_ct_new;

            delta = 0.0;

            // Obtengo las tareas sorteadas.
            task_x = (int)(random1 * INPUT.tasks_count);
            task_y = (int)(random2 * INPUT.tasks_count);

            if (task_x != task_y) {
                // Obtengo las máquinas a las que estan asignadas las tareas.
                machine_a = EA_THREADS[thread_id].population[solution_index].task_assignment[task_x]; // Máquina a.
                machine_b = EA_THREADS[thread_id].population[solution_index].task_assignment[task_y]; // Máquina b.

                if (machine_a != machine_b) {
                    // Calculo el delta del swap sorteado.

                    // Máquina 1.
                    machine_a_ct_old = EA_THREADS[thread_id].population[solution_index].machine_compute_time[machine_a];

                    machine_a_ct_new = machine_a_ct_old;
                    machine_a_ct_new = machine_a_ct_new - get_etc_value(machine_a, task_x); // Resto del ETC de x en a.
                    machine_a_ct_new = machine_a_ct_new + get_etc_value(machine_a, task_y); // Sumo el ETC de y en a.

                    // Máquina 2.
                    machine_b_ct_old = EA_THREADS[thread_id].population[solution_index].machine_compute_time[machine_b];

                    machine_b_ct_new = machine_b_ct_old;
                    machine_b_ct_new = machine_b_ct_new - get_etc_value(machine_b, task_y); // Resto el ETC de y en b.
                    machine_b_ct_new = machine_b_ct_new + get_etc_value(machine_b, task_x); // Sumo el ETC de x en b.

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
                    #endif
                }
            }
            if ((loop == 0) || (best_delta > delta)) {
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

            if ((loop == 0) || (best_delta > delta)) {
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

    if (best_delta > 0) {
        int src_machine;
        src_machine = EA_THREADS[thread_id].population[solution_index].task_assignment[best_src_task];
        
        int dst_machine;

        if (best_movimiento == PALS_MOVIMIENTO_SWAP) {
            dst_machine = EA_THREADS[thread_id].population[solution_index].task_assignment[best_dst];

            /*
            #ifdef DEBUG_3
                fprintf(stderr, "[DEBUG] <thread=%d> PALS swap task %d in %d with task %d in %d\n", thread_id, src_task, src_machine, best_dst_task, dst_machine);
            #endif
            */

            /* Hago el swap */
            EA_THREADS[thread_id].population[solution_index].task_assignment[best_dst] = src_machine;
            EA_THREADS[thread_id].population[solution_index].task_assignment[best_src_task] = dst_machine;

            /* Actualizo el compute time de las maquinas involucradas */
            /*EA_THREADS[thread_id].population[solution_index].machine_compute_time[dst_machine] += get_etc_value(dst_machine, best_src_task);
            EA_THREADS[thread_id].population[solution_index].machine_compute_time[dst_machine] -= get_etc_value(dst_machine, best_dst);

            EA_THREADS[thread_id].population[solution_index].machine_compute_time[src_machine] += get_etc_value(src_machine, best_dst);
            EA_THREADS[thread_id].population[solution_index].machine_compute_time[src_machine] -= get_etc_value(src_machine, best_src_task);*/

        } else if (best_movimiento == PALS_MOVIMIENTO_MOVE) {
            dst_machine = best_dst;
            
            /* Hago el move */
            EA_THREADS[thread_id].population[solution_index].task_assignment[best_src_task] = dst_machine;

            /* Actualizo el compute time de las maquinas involucradas */
            /*EA_THREADS[thread_id].population[solution_index].machine_compute_time[dst_machine] += get_etc_value(dst_machine, best_src_task);
            EA_THREADS[thread_id].population[solution_index].machine_compute_time[src_machine] -= get_etc_value(src_machine, best_src_task);*/
        } else {
            assert(1 == 0);
            dst_machine = -1;
        }

        /* Actualizo el makespan general del schedule */
        /*int recompute_energy;
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

        if (recompute_energy == 1) {*/
            // Recalculo todo

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
        /*} else {
            // Solo actualizo las máquinas involucradas

            if (best_movimiento == PALS_MOVIMIENTO_SWAP) {
                EA_THREADS[thread_id].population[solution_index].machine_energy_consumption[src_machine] +=
                    get_energy_value(src_machine, best_dst) - get_energy_value(src_machine, best_src_task);

                EA_THREADS[thread_id].population[solution_index].machine_energy_consumption[dst_machine] +=
                    get_energy_value(dst_machine, best_src_task) - get_energy_value(dst_machine, best_dst);

            } else if (best_movimiento == PALS_MOVIMIENTO_MOVE) {
                EA_THREADS[thread_id].population[solution_index].machine_energy_consumption[src_machine] -=
                    get_energy_value(src_machine, best_src_task);

                EA_THREADS[thread_id].population[solution_index].machine_energy_consumption[dst_machine] +=
                    get_energy_value(dst_machine, best_src_task);

            } else {
                assert(1 == 0);
            }
        }*/
    }

    refresh_solution(&EA_THREADS[thread_id].population[solution_index]);

    /*
    #ifdef DEBUG_3
        FLOAT aux_makespan, aux_energy;
        aux_makespan = EA_THREADS[thread_id].population[solution_index].makespan;
        aux_energy = EA_THREADS[thread_id].population[solution_index].energy_consumption;
    
        refresh_solution(&EA_THREADS[thread_id].population[solution_index]);
        
        fprintf(stderr, "[INFO] makespan=%.2f/%.2f energy=%.2f/%.2f\n", 
            aux_makespan, EA_THREADS[thread_id].population[solution_index].makespan,
            aux_energy, EA_THREADS[thread_id].population[solution_index].energy_consumption);
        
        assert(aux_makespan == EA_THREADS[thread_id].population[solution_index].makespan);
        assert(aux_energy == EA_THREADS[thread_id].population[solution_index].energy_consumption);
    #endif*/

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
