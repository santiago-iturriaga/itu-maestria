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

struct ls_movement movements[MAX_THREADS][PALS__MAX_INTENTOS];

struct pals_aux_state {
    int **mod_machines;
};

struct pals_aux_state state;

void pals_init(int thread_id) {
    state.mod_machines = (int**)(malloc(sizeof(int*) * INPUT.thread_count));
    for (int i = 0; i < INPUT.thread_count; i++) {
        state.mod_machines[i] = (int*)(malloc(sizeof(int) * INPUT.machines_count));
        memset(state.mod_machines[i], 0, sizeof(int) * INPUT.machines_count);
    }
}

void pals_free(int thread_id) {
    for (int i = 0; i < INPUT.thread_count; i++) {
        free(state.mod_machines[i]);
    }
    free(state.mod_machines);
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
    
    int selected_machine;
    selected_machine = max_et_machine;
    //selected_machine = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]) * INPUT.tasks_count;

    FLOAT score;
    int movimiento;

    FLOAT best_score = VERY_BIG_FLOAT;
    int best_index = -1;

    FLOAT random1, random2;
    FLOAT current_makespan = EA_THREADS[thread_id].population[solution_index].makespan;

    for (int loop = 0; loop < PALS__MAX_INTENTOS; loop++) {       
        if (RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]) < PALS__SWAP_SEARCH) {
            movimiento = PALS_MOVIMIENTO_SWAP;  
        } else {
            movimiento = PALS_MOVIMIENTO_MOVE;
        }

        if (movimiento == PALS_MOVIMIENTO_SWAP) {
            // =================
            // Swap
            int task_x, task_y;
            int machine_a, machine_b;

            FLOAT machine_a_ct_old, machine_b_ct_old;
            FLOAT machine_a_ct_new, machine_b_ct_new;

            score = 0.0;

            // Obtengo las tareas sorteadas.
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
                    task_x, task_y, EA_THREADS[thread_id].population[solution_index].makespan);
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
                if ((machine_a_ct_new > current_makespan) || (machine_b_ct_new > current_makespan)) {
                    // Luego del movimiento aumenta el makespan. Intento desestimularlo lo más posible.
                    if (machine_a_ct_new > current_makespan) score = score + (machine_a_ct_new - current_makespan);
                    if (machine_b_ct_new > current_makespan) score = score + (machine_b_ct_new - current_makespan);
                } else if ((machine_a_ct_old+1 >= current_makespan) || (machine_b_ct_old+1 >= current_makespan)) {
                    // Antes del movimiento una las de máquinas definía el makespan. Estos son los mejores movimientos.
                    if (machine_a_ct_old+1 >= current_makespan) {
                        score = score + (machine_a_ct_new - machine_a_ct_old);
                    } else {
                        score = score + 1 / (machine_a_ct_new - machine_a_ct_old);
                    }

                    if (machine_b_ct_old+1 >= current_makespan) {
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

            movements[thread_id][loop].score = score;
            movements[thread_id][loop].tipo = PALS_MOVIMIENTO_SWAP;
            movements[thread_id][loop].src_task = task_x;
            movements[thread_id][loop].dst = task_y;

            if (best_score > score) {
                best_score = score;
                best_index = loop;
            }
        } else if (movimiento == PALS_MOVIMIENTO_MOVE) {
            // =================
            // Move

            int task_x;
            int machine_a, machine_b;

            float machine_a_ct_old, machine_b_ct_old;
            float machine_a_ct_new, machine_b_ct_new;

            score = 0.0;

            // ================= Obtengo la tarea sorteada
            
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
                if (machine_b_ct_new > current_makespan) {
                    // Luego del movimiento aumenta el makespan. Intento desestimularlo lo más posible.
                    score = score + (machine_b_ct_new - current_makespan);
                } else if (machine_a_ct_old+1 >= current_makespan) {
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

            movements[thread_id][loop].score = score;
            movements[thread_id][loop].tipo = PALS_MOVIMIENTO_MOVE;
            movements[thread_id][loop].src_task = task_x;
            movements[thread_id][loop].dst = machine_b;

            if (best_score > score) {
                best_score = score;
                best_index = loop;
            }
        }
        
        #ifdef DEBUG_3
            fprintf(stderr, "[DEBUG] Delta %.2f\n", score);
        #endif
    }

    #ifdef DEBUG_3
        fprintf(stderr, "[DEBUG] best_score = %.2f\n", best_score);
    #endif

    if (best_score < 0) {
        int current_index = best_index;
        int task_src;
        int task_dst;
        int machine_src;
        int machine_dst;
        
        for (int i = 0; i < PALS__MAX_INTENTOS; i++) {
            if (movements[thread_id][current_index].score < 0) {
                task_src = movements[thread_id][current_index].src_task;
                machine_src = EA_THREADS[thread_id].population[solution_index].task_assignment[task_src];

                if (state.mod_machines[thread_id][machine_src] == 0) {
                    if (movements[thread_id][current_index].tipo == PALS_MOVIMIENTO_SWAP) {
                        task_dst = movements[thread_id][current_index].dst;
                        machine_dst = EA_THREADS[thread_id].population[solution_index].task_assignment[task_dst];

                        if (state.mod_machines[thread_id][machine_dst] == 0) {
                            /* Hago el swap */
                            EA_THREADS[thread_id].population[solution_index].task_assignment[task_dst] = machine_src;
                            EA_THREADS[thread_id].population[solution_index].task_assignment[task_src] = machine_dst;
                            
                            state.mod_machines[thread_id][machine_src] = 1;
                            state.mod_machines[thread_id][machine_dst] = 1;
                        }
                    } else if (movements[thread_id][current_index].tipo == PALS_MOVIMIENTO_MOVE) {
                        machine_dst = movements[thread_id][current_index].dst;
                        
                        if (state.mod_machines[thread_id][machine_dst] == 0) {
                            /* Hago el move */
                            EA_THREADS[thread_id].population[solution_index].task_assignment[task_src] = machine_dst;
                            
                            state.mod_machines[thread_id][machine_src] = 1;
                            state.mod_machines[thread_id][machine_dst] = 1;
                        }
                    } else {
                        assert(1 == 0);
                    }
                }
                
                current_index++;
                if (current_index == PALS__MAX_INTENTOS) current_index = 0;
            }
        }
    }

    memset(state.mod_machines[thread_id], 0, sizeof(int) * INPUT.machines_count);
    
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

        if (fitness_post < fitness_pre) {
            CHC_PALS_COUNT_FITNESS_IMPROV[thread_id]++;
            
            /*if (best_movimiento == PALS_MOVIMIENTO_SWAP) {
                CHC_PALS_COUNT_FITNESS_IMPROV_SWAP[thread_id]++;
            } else if (best_movimiento == PALS_MOVIMIENTO_MOVE) {
                CHC_PALS_COUNT_FITNESS_IMPROV_MOVE[thread_id]++;
            }*/
        }
        if (fitness_post > fitness_pre) {
            CHC_PALS_COUNT_FITNESS_DECLINE[thread_id]++;
            
            /*if (best_movimiento == PALS_MOVIMIENTO_SWAP) {
                CHC_PALS_COUNT_FITNESS_DECLINE_SWAP[thread_id]++;
            } else if (best_movimiento == PALS_MOVIMIENTO_MOVE) {
                CHC_PALS_COUNT_FITNESS_DECLINE_MOVE[thread_id]++;
            }*/
        }
    #endif
}
