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
    int CHC_PALS_COUNT_FITNESS_IMPROV_SWAP[MAX_THREADS] = {0};
    int CHC_PALS_COUNT_FITNESS_IMPROV_MOVE[MAX_THREADS] = {0};
    int CHC_PALS_COUNT_FITNESS_DECLINE[MAX_THREADS] = {0};
    int CHC_PALS_COUNT_FITNESS_DECLINE_SWAP[MAX_THREADS] = {0};
    int CHC_PALS_COUNT_FITNESS_DECLINE_MOVE[MAX_THREADS] = {0};
#endif

#define PALS_MOVIMIENTO_SWAP 0
#define PALS_MOVIMIENTO_MOVE 1

/* LS */
struct ls_movement {
    FLOAT score;
    int tipo;
    int src_task;
    int dst;
};

struct ls_movement movements[MAX_THREADS];

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
    FLOAT makespan = EA_THREADS[thread_id].population[solution_index].machine_compute_time[0];
    int makespan_machine_index = 0;
    
    FLOAT energy_machine = EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[0];
    int energy_machine_index = 0;
    
    for (int m = 1; m < INPUT.machines_count; m++) {
        if (EA_THREADS[thread_id].population[solution_index].machine_compute_time[m] > makespan) {
            makespan_machine_index = m;
            makespan = EA_THREADS[thread_id].population[solution_index].machine_compute_time[m];
            
        } else if (EA_THREADS[thread_id].population[solution_index].machine_compute_time[m] == makespan) {
            if (RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]) > 0.5) {
                makespan_machine_index = m;
                makespan = EA_THREADS[thread_id].population[solution_index].machine_compute_time[m];
            }
        }

        if (EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[m] > energy_machine) {
            energy_machine_index = m;
            energy_machine = EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[m];
            
        } else if (EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[m] == energy_machine) {
            if (RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]) > 0.5) {
                energy_machine_index = m;
                energy_machine = EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[m];
            }
        }
    }
    
    for (int i = 0; i < PALS__MAX_BUSQUEDAS; i++) {       
        int selected_machine;
        if (RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]) < 0.33) {
            selected_machine = (int)(RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]) * INPUT.tasks_count);
        } else if (RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]) < 0.66) {
            selected_machine = makespan_machine_index;
        } else {
            selected_machine = energy_machine_index;
        }
        

        FLOAT score;
        int movimiento;
        
        movements[thread_id].score = VERY_BIG_FLOAT;
        movements[thread_id].tipo = -1;
        movements[thread_id].src_task = -1;
        movements[thread_id].dst = -1;

        FLOAT random1, random2;

        // Obtengo las tareas sorteadas.
        int task_x;
        int machine_a;
        
        random1 = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]);
        task_x = (int)(random1 * INPUT.tasks_count);
        machine_a = EA_THREADS[thread_id].population[solution_index].task_assignment[task_x]; // Máquina a.
        
        if (selected_machine != -1) {
            for (int i = 0; (i < INPUT.tasks_count) && (machine_a != selected_machine); i++) {
                task_x++;
                if (task_x == INPUT.tasks_count) task_x = 0;
                
                machine_a = EA_THREADS[thread_id].population[solution_index].task_assignment[task_x]; // Máquina a.
            }
        }

        for (int loop = 0; loop < PALS__MAX_INTENTOS; loop++) {       
            if (RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]) < PALS__SWAP_SEARCH) {
                movimiento = PALS_MOVIMIENTO_SWAP;  
            } else {
                movimiento = PALS_MOVIMIENTO_MOVE;
            }

            if (movimiento == PALS_MOVIMIENTO_SWAP) {
                // =================
                // Swap
                int task_y;
                int machine_b;

                FLOAT machine_a_ct_old, machine_b_ct_old;
                FLOAT machine_a_ct_new, machine_b_ct_new;

                score = 0.0;
           
                random2 = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]);    
                task_y = (int)(random2 * INPUT.tasks_count);
                machine_b = EA_THREADS[thread_id].population[solution_index].task_assignment[task_y]; // Máquina b.
                
                for (int i = 0; (i < INPUT.tasks_count) && (machine_a == machine_b); i++) {
                    task_y++;
                    if (task_y == INPUT.tasks_count) task_y = 0;
                    
                    machine_b = EA_THREADS[thread_id].population[solution_index].task_assignment[task_y]; // Máquina b.
                }

                // Calculo el score del swap sorteado.

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
                        task_x, task_y, makespan);
                    fprintf(stderr, "[DEBUG] Machine CT (a) %.2f to %.2f (b) %.2f to %.2f\n", 
                        machine_a_ct_old, machine_a_ct_new,
                        machine_b_ct_old, machine_b_ct_new);
                #endif

                FLOAT max_old;
                max_old = machine_a_ct_old;
                if (max_old < machine_b_ct_old) max_old = machine_b_ct_old;

                #if defined(PALS__SIMPLE_DELTA)
                    if ((machine_a_ct_new > max_old) || (machine_b_ct_new > max_old)) {
                        score = VERY_BIG_FLOAT - (max_old - machine_a_ct_new) + (max_old - machine_b_ct_new);
                    } else {
                        score = (machine_a_ct_new - max_old) + (machine_b_ct_new - max_old);
                    }
                #endif
                #if defined(PALS__COMPLEX_DELTA)
                    if ((machine_a_ct_new > makespan) || (machine_b_ct_new > makespan)) {
                        // Luego del movimiento aumenta el makespan. Intento desestimularlo lo más posible.
                        if (machine_a_ct_new > makespan) score = score + (machine_a_ct_new - makespan);
                        if (machine_b_ct_new > makespan) score = score + (machine_b_ct_new - makespan);
                    } else if ((machine_a_ct_old+1 >= makespan) || (machine_b_ct_old+1 >= makespan)) {
                        // Antes del movimiento una las de máquinas definía el makespan. Estos son los mejores movimientos.
                        if (machine_a_ct_old+1 >= makespan) {
                            score = score + (machine_a_ct_new - machine_a_ct_old);
                        } else {
                            score = score + 1 / (machine_a_ct_new - machine_a_ct_old);
                        }

                        if (machine_b_ct_old+1 >= makespan) {
                            score = score + (machine_b_ct_new - machine_b_ct_old);
                        } else {
                            score = score + 1 / (machine_b_ct_new - machine_b_ct_old);
                        }
                    } else {
                        // Ninguna de las máquinas intervenía en el makespan. Intento favorecer lo otros movimientos.
                        score = score + (machine_a_ct_new - machine_a_ct_old);
                        score = score + (machine_b_ct_new - machine_b_ct_old);
                        score = 1 / score;
                    }
                #endif

                if (movements[thread_id].score > score) {
                    movements[thread_id].score = score;
                    movements[thread_id].tipo = PALS_MOVIMIENTO_SWAP;
                    movements[thread_id].src_task = task_x;
                    movements[thread_id].dst = task_y;
                }
            } else if (movimiento == PALS_MOVIMIENTO_MOVE) {
                // =================
                // Move

                int machine_b;

                float machine_a_ct_old, machine_b_ct_old;
                float machine_a_ct_new, machine_b_ct_new;

                score = 0.0;

                // ================= Obtengo la tarea sorteada
                random2 = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]);    
                machine_b = (int)(random2 * INPUT.machines_count);
                
                for (int i = 0; (i < INPUT.machines_count) && (machine_a == machine_b); i++) {
                    machine_b++;
                    if (machine_b == INPUT.machines_count) machine_b = 0;
                }

                machine_a_ct_old = EA_THREADS[thread_id].population[solution_index].machine_compute_time[machine_a];
                machine_b_ct_old = EA_THREADS[thread_id].population[solution_index].machine_compute_time[machine_b];

                // Calculo el score del swap sorteado.
                machine_a_ct_new = machine_a_ct_old - get_etc_value(machine_a, task_x); // Resto del ETC de x en a.
                machine_b_ct_new = machine_b_ct_old + get_etc_value(machine_b, task_x); // Sumo el ETC de x en b.

                #if defined(PALS__SIMPLE_DELTA)
                    float max_old;
                    max_old = machine_a_ct_old;
                    if (max_old < machine_b_ct_old) max_old = machine_b_ct_old;

                    if ((machine_a_ct_new > max_old) || (machine_b_ct_new > max_old)) {
                        score = VERY_BIG_FLOAT - (max_old - machine_a_ct_new) + (max_old - machine_b_ct_new);
                    } else {
                        score = (machine_a_ct_new - max_old) + (machine_b_ct_new - max_old);
                    }
                #endif
                #if defined(PALS__COMPLEX_DELTA)
                    if (machine_b_ct_new > makespan) {
                        // Luego del movimiento aumenta el makespan. Intento desestimularlo lo más posible.
                        score = score + (machine_b_ct_new - makespan);
                    } else if (machine_a_ct_old+1 >= makespan) {
                        // Antes del movimiento una las de máquinas definía el makespan. Estos son los mejores movimientos.
                        score = score + (machine_a_ct_new - machine_a_ct_old);
                        score = score + 1 / (machine_b_ct_new - machine_b_ct_old);
                    } else {
                        // Ninguna de las máquinas intervenía en el makespan. Intento favorecer lo otros movimientos.
                        score = score + (machine_a_ct_new - machine_a_ct_old);
                        score = score + (machine_b_ct_new - machine_b_ct_old);
                        score = 1 / score;
                    }
                #endif
                
                if (movements[thread_id].score > score) {
                    movements[thread_id].score = score;
                    movements[thread_id].tipo = PALS_MOVIMIENTO_MOVE;
                    movements[thread_id].src_task = task_x;
                    movements[thread_id].dst = machine_b;
                }
            }
            
            #ifdef DEBUG_3
                fprintf(stderr, "[DEBUG] delta %.2f\n", score);
            #endif
        }

        #ifdef DEBUG_3
            fprintf(stderr, "[DEBUG] best_score = %.2f\n", movements[thread_id].score);
        #endif

        if (movements[thread_id].score < 0) {
            int task_src;
            int task_dst;
            int machine_src;
            int machine_dst;
            
            task_src = movements[thread_id].src_task;
            machine_src = EA_THREADS[thread_id].population[solution_index].task_assignment[task_src];

            if (movements[thread_id].tipo == PALS_MOVIMIENTO_SWAP) {
                task_dst = movements[thread_id].dst;
                machine_dst = EA_THREADS[thread_id].population[solution_index].task_assignment[task_dst];

                /* Hago el swap */
                EA_THREADS[thread_id].population[solution_index].task_assignment[task_dst] = machine_src;
                EA_THREADS[thread_id].population[solution_index].task_assignment[task_src] = machine_dst;
                
                EA_THREADS[thread_id].population[solution_index].machine_compute_time[machine_src] =
                    EA_THREADS[thread_id].population[solution_index].machine_compute_time[machine_src] +
                    get_etc_value(machine_src, task_dst) - get_etc_value(machine_src, task_src);

                EA_THREADS[thread_id].population[solution_index].machine_compute_time[machine_dst] =
                    EA_THREADS[thread_id].population[solution_index].machine_compute_time[machine_dst] +
                    get_etc_value(machine_dst, task_src) - get_etc_value(machine_dst, task_dst);
                    
                EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[machine_src] = 
                    EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[machine_src] +
                    (get_scenario_energy_max(machine_src) * get_etc_value(machine_src, task_dst)) -
                    (get_scenario_energy_max(machine_src) * get_etc_value(machine_src, task_src));
                    
                EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[machine_dst] = 
                    EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[machine_dst] +
                    (get_scenario_energy_max(machine_dst) * get_etc_value(machine_dst, task_src)) -
                    (get_scenario_energy_max(machine_dst) * get_etc_value(machine_dst, task_dst));
                    
            } else if (movements[thread_id].tipo == PALS_MOVIMIENTO_MOVE) {
                machine_dst = movements[thread_id].dst;
                
                /* Hago el move */
                EA_THREADS[thread_id].population[solution_index].task_assignment[task_src] = machine_dst;

                EA_THREADS[thread_id].population[solution_index].machine_compute_time[machine_src] -= get_etc_value(machine_src, task_src);
                EA_THREADS[thread_id].population[solution_index].machine_compute_time[machine_dst] += get_etc_value(machine_dst, task_src);

                EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[machine_src] -= 
                    get_scenario_energy_max(machine_src) * get_etc_value(machine_src, task_src);
                EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[machine_dst] += 
                    get_scenario_energy_max(machine_dst) * get_etc_value(machine_dst, task_src);
                
            } else {
                assert(1 == 0);
            }
        }
        
        // Recalculo makespan y energy
        /*if (i + 1 < PALS__MAX_BUSQUEDAS) {
            makespan = EA_THREADS[thread_id].population[solution_index].machine_compute_time[0];
            makespan_machine_index = 0;
            
            energy_machine = EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[0];
            energy_machine_index = 0;
            
            for (int m = 1; m < INPUT.machines_count; m++) {
                if (EA_THREADS[thread_id].population[solution_index].machine_compute_time[m] > makespan) {
                    makespan_machine_index = m;
                    makespan = EA_THREADS[thread_id].population[solution_index].machine_compute_time[m];
                    
                } else if (EA_THREADS[thread_id].population[solution_index].machine_compute_time[m] == makespan) {
                    if (RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]) > 0.5) {
                        makespan_machine_index = m;
                        makespan = EA_THREADS[thread_id].population[solution_index].machine_compute_time[m];
                    }
                }

                if (EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[m] > energy_machine) {
                    energy_machine_index = m;
                    energy_machine = EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[m];
                    
                } else if (EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[m] == energy_machine) {
                    if (RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]) > 0.5) {
                        energy_machine_index = m;
                        energy_machine = EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[m];
                    }
                }
            }
        }*/
    }
    
    recompute_metrics(&EA_THREADS[thread_id].population[solution_index]);
    
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

        if (fitness_post < fitness_pre) {
            CHC_PALS_COUNT_FITNESS_IMPROV[thread_id]++;
            
            if (movements[thread_id].tipo == PALS_MOVIMIENTO_SWAP) {
                CHC_PALS_COUNT_FITNESS_IMPROV_SWAP[thread_id]++;
            } else if (movements[thread_id].tipo == PALS_MOVIMIENTO_MOVE) {
                CHC_PALS_COUNT_FITNESS_IMPROV_MOVE[thread_id]++;
            }
        }
        if (fitness_post > fitness_pre) {
            CHC_PALS_COUNT_FITNESS_DECLINE[thread_id]++;
            
            if (movements[thread_id].tipo == PALS_MOVIMIENTO_SWAP) {
                CHC_PALS_COUNT_FITNESS_DECLINE_SWAP[thread_id]++;
            } else if (movements[thread_id].tipo == PALS_MOVIMIENTO_MOVE) {
                CHC_PALS_COUNT_FITNESS_DECLINE_MOVE[thread_id]++;
            }
        }
    #endif
}
