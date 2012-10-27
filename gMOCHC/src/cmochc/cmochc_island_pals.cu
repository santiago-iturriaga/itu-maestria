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
    int CHC_PALS_COUNT_FITNESS_IMPROV_RANDOM[MAX_THREADS] = {0};
    int CHC_PALS_COUNT_FITNESS_IMPROV_MAKESPAN[MAX_THREADS] = {0};
    int CHC_PALS_COUNT_FITNESS_IMPROV_ENERGY[MAX_THREADS] = {0};
    int CHC_PALS_COUNT_FITNESS_DECLINE[MAX_THREADS] = {0};
    int CHC_PALS_COUNT_FITNESS_DECLINE_SWAP[MAX_THREADS] = {0};
    int CHC_PALS_COUNT_FITNESS_DECLINE_MOVE[MAX_THREADS] = {0};
#endif

#define PALS__MOVIMIENTO_SWAP 0
#define PALS__MOVIMIENTO_MOVE 1

//#define MAX__COLLECTED_TASKS 16
#define MAX__COLLECTED_TASKS 32
//#define MAX__COLLECTED_TASKS 64

#define PALS__MAKESPAN_SEARCH 0
#define PALS__ENERGY_SEARCH 1

#define PALS__MAKESPAN_EPSILON 1

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

inline FLOAT compute_movement_score(int thread_id, int search_type,
    FLOAT worst_compute_time, FLOAT worst_energy, 
    FLOAT machine_a_ct_new, FLOAT machine_a_ct_old,
    FLOAT machine_b_ct_new, FLOAT machine_b_ct_old,
    FLOAT machine_a_en_new, FLOAT machine_a_en_old,
    FLOAT machine_b_en_new, FLOAT machine_b_en_old) {
        
    FLOAT score = 0;
        
    #if defined(PALS__SIMPLE_FITNESS_0)
        if ((machine_a_ct_new > worst_compute_time) || (machine_b_ct_new > worst_compute_time) ||
            (machine_a_en_new > worst_energy) || (machine_b_en_new > worst_energy)) {

            score = VERY_BIG_FLOAT;
        } else {
            score = ((machine_a_ct_new - machine_a_ct_old) / worst_compute_time +
                    (machine_b_ct_new - machine_b_ct_old) / worst_compute_time) * EA_THREADS[thread_id].weight_makespan;

            score += ((machine_a_en_new - machine_a_en_old) / worst_energy +
                    (machine_b_en_new - machine_b_en_old) / worst_energy) * EA_THREADS[thread_id].weight_energy;
        }
    #endif
    #if defined(PALS__SIMPLE_FITNESS_1)
        if ((machine_a_ct_new > worst_compute_time) || (machine_b_ct_new > worst_compute_time)) {

            score = VERY_BIG_FLOAT;
        } else {
            score = ((machine_a_ct_new - machine_a_ct_old) / worst_compute_time +
                    (machine_b_ct_new - machine_b_ct_old) / worst_compute_time) * EA_THREADS[thread_id].weight_makespan;

            score += ((machine_a_en_new - machine_a_en_old) / worst_energy +
                    (machine_b_en_new - machine_b_en_old) / worst_energy) * EA_THREADS[thread_id].weight_energy;
        }
    #endif
    #if defined(PALS__SIMPLE_FITNESS_2)
        if (search_type == PALS_MAKESPAN_SEARCH) {
            if ((machine_a_ct_new > worst_compute_time) || (machine_b_ct_new > worst_compute_time)) {
                score = VERY_BIG_FLOAT;
            } else {
                if ((machine_a_ct_old + PALS_MAKESPAN_EPSILON >= worst_compute_time) || (machine_b_ct_old + PALS_MAKESPAN_EPSILON >= worst_compute_time)) {
                    score = ((machine_a_ct_new - machine_a_ct_old) / worst_compute_time +
                            (machine_b_ct_new - machine_b_ct_old) / worst_compute_time) * EA_THREADS[thread_id].weight_makespan;

                    score += ((machine_a_en_new - machine_a_en_old) / machine_a_en_old +
                            (machine_b_en_new - machine_b_en_old) / machine_b_en_old) * EA_THREADS[thread_id].weight_energy;
                } else {
                    score = ((machine_a_ct_new - machine_a_ct_old) / worst_compute_time +
                            (machine_b_ct_new - machine_b_ct_old) / worst_compute_time) * EA_THREADS[thread_id].weight_makespan;

                    score += ((machine_a_en_new - machine_a_en_old) / machine_a_en_old +
                            (machine_b_en_new - machine_b_en_old) / machine_b_en_old) * EA_THREADS[thread_id].weight_energy;
                            
                    score = 1 / score;
                }
            }
        } else if (search_type == PALS_ENERGY_SEARCH) {
            if (machine_a_en_new - machine_a_en_old + machine_b_en_new - machine_b_en_old > 0) {
                score = VERY_BIG_FLOAT;
            } else {
                score = ((machine_a_ct_new - machine_a_ct_old) / worst_compute_time +
                        (machine_b_ct_new - machine_b_ct_old) / worst_compute_time) * EA_THREADS[thread_id].weight_makespan;

                score += ((machine_a_en_new - machine_a_en_old) / machine_a_en_old +
                        (machine_b_en_new - machine_b_en_old) / machine_b_en_old) * EA_THREADS[thread_id].weight_energy;
            }
        } else {
            if ((machine_a_ct_new > worst_compute_time) || (machine_b_ct_new > worst_compute_time) ||
                (machine_a_en_new - machine_a_en_old + machine_b_en_new - machine_b_en_old > 0)) {
                score = VERY_BIG_FLOAT;
            } else {
                score = ((machine_a_ct_new - machine_a_ct_old) / worst_compute_time +
                        (machine_b_ct_new - machine_b_ct_old) / worst_compute_time) * EA_THREADS[thread_id].weight_makespan;

                score += ((machine_a_en_new - machine_a_en_old) / machine_a_en_old +
                        (machine_b_en_new - machine_b_en_old) / machine_b_en_old) * EA_THREADS[thread_id].weight_energy;
            }
        }
    #endif
    #if defined(PALS__SIMPLE_FITNESS_3)
        if (search_type == PALS__MAKESPAN_SEARCH) {
            if ((machine_a_ct_new > worst_compute_time) || (machine_b_ct_new > worst_compute_time)) {
                score = VERY_BIG_FLOAT;
            } else {
                if ((machine_a_ct_old + PALS__MAKESPAN_EPSILON >= worst_compute_time) || (machine_b_ct_old + PALS__MAKESPAN_EPSILON >= worst_compute_time)) {
                    if (machine_a_ct_old + PALS__MAKESPAN_EPSILON >= worst_compute_time) {
                        score = (machine_a_ct_new - machine_a_ct_old) - machine_a_ct_old;
                    } else {
                        score = (((machine_a_ct_new - machine_a_ct_old) / machine_a_ct_old) * EA_THREADS[thread_id].weight_makespan);
                    }
                    
                    if (machine_b_ct_old + PALS__MAKESPAN_EPSILON >= worst_compute_time) {
                        score += (machine_b_ct_new - machine_b_ct_old) - machine_a_ct_old;
                    } else {
                        score += (((machine_b_ct_new - machine_b_ct_old) / machine_a_ct_old) * EA_THREADS[thread_id].weight_energy);
                    }
                    
                    score += (((machine_a_en_new - machine_a_en_old) / machine_a_en_old) * EA_THREADS[thread_id].weight_energy);
                    score += (((machine_b_en_new - machine_b_en_old) / machine_b_en_old) * EA_THREADS[thread_id].weight_energy);
                } else {
                    score = (((machine_a_ct_new - machine_a_ct_old) / machine_a_ct_old) * EA_THREADS[thread_id].weight_makespan);
                    score += (((machine_b_ct_new - machine_b_ct_old) / machine_b_ct_old) * EA_THREADS[thread_id].weight_makespan);
                            
                    score += (((machine_a_en_new - machine_a_en_old) / machine_a_en_old) * EA_THREADS[thread_id].weight_energy);
                    score += (((machine_b_en_new - machine_b_en_old) / machine_b_en_old) * EA_THREADS[thread_id].weight_energy);
                }
            }
        } else if (search_type == PALS__ENERGY_SEARCH) {
            if (machine_a_en_new - machine_a_en_old + machine_b_en_new - machine_b_en_old > 0) {
                score = VERY_BIG_FLOAT;
            } else {
                score = ((machine_a_ct_new - machine_a_ct_old) / machine_a_ct_old) * EA_THREADS[thread_id].weight_makespan;
                score += ((machine_b_ct_new - machine_b_ct_old) / machine_b_ct_old) * EA_THREADS[thread_id].weight_makespan;
                        
                score += ((machine_a_en_new - machine_a_en_old) / machine_a_en_old) * EA_THREADS[thread_id].weight_energy;
                score += ((machine_b_en_new - machine_b_en_old) / machine_b_en_old) * EA_THREADS[thread_id].weight_energy;
            }
        }
    #endif
    #if defined(PALS__SIMPLE_DELTA)
        if ((machine_a_ct_new > max_old) || (machine_b_ct_new > max_old)) {
            score = VERY_BIG_FLOAT - (max_old - machine_a_ct_new) + (max_old - machine_b_ct_new);
        } else {
            score = (machine_a_ct_new - max_old) + (machine_b_ct_new - max_old);
        }
    #endif
    #if defined(PALS__COMPLEX_DELTA)
        if ((machine_a_ct_new > worst_compute_time) || (machine_b_ct_new > worst_compute_time)) {
            // Luego del movimiento aumenta el worst_compute_time. Intento desestimularlo lo más posible.
            if (machine_a_ct_new > worst_compute_time) score = score + (machine_a_ct_new - worst_compute_time);
            if (machine_b_ct_new > worst_compute_time) score = score + (machine_b_ct_new - worst_compute_time);
        } else if ((machine_a_ct_old+1 >= worst_compute_time) || (machine_b_ct_old+1 >= worst_compute_time)) {
            // Antes del movimiento una las de máquinas definía el worst_compute_time. Estos son los mejores movimientos.
            if (machine_a_ct_old+1 >= worst_compute_time) {
                score = score + (machine_a_ct_new - machine_a_ct_old);
            } else {
                score = score + 1 / (machine_a_ct_new - machine_a_ct_old);
            }

            if (machine_b_ct_old+1 >= worst_compute_time) {
                score = score + (machine_b_ct_new - machine_b_ct_old);
            } else {
                score = score + 1 / (machine_b_ct_new - machine_b_ct_old);
            }
        } else {
            // Ninguna de las máquinas intervenía en el worst_compute_time. Intento favorecer lo otros movimientos.
            score = score + (machine_a_ct_new - machine_a_ct_old);
            score = score + (machine_b_ct_new - machine_b_ct_old);
            score = 1 / score;
        }
    #endif
    
    return score;
}

int pals_search(int thread_id, int solution_index) {
    int worst_compute_time_machine_tasks[MAX__COLLECTED_TASKS];
    int worst_compute_time_machine_count;

    int worst_energy_machine_tasks[MAX__COLLECTED_TASKS];
    int worst_energy_machine_count;

    int less_compute_time_machine_tasks[MAX__COLLECTED_TASKS];
    int less_compute_time_machine_count;

    int less_energy_machine_tasks[MAX__COLLECTED_TASKS];
    int less_energy_machine_count;

    FLOAT makespan_pre = EA_THREADS[thread_id].population[solution_index].makespan;
    FLOAT energy_pre = EA_THREADS[thread_id].population[solution_index].energy_consumption;

    #ifdef DEBUG_1
        CHC_PALS_COUNT_EXECUTIONS[thread_id]++;
    #endif

    int search_type;
    int count_movements = 0;

    int task_x = 0;
    int machine_a = 0;
    
    FLOAT score;
    int movimiento;

    FLOAT random, random1, random2;

    // Busco la máquina con mayor compute time.
    FLOAT worst_compute_time = EA_THREADS[thread_id].population[solution_index].machine_compute_time[0];
    int worst_compute_time_index = 0;

    FLOAT less_compute_time = EA_THREADS[thread_id].population[solution_index].machine_compute_time[0];
    int less_compute_time_index = 0;

    // Busco la máquina con mayor consumo de energy.
    FLOAT worst_energy = EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[0];
    int worst_energy_index = 0;

    FLOAT less_energy = EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[0];
    int less_energy_index = 0;

    for (int m = 1; m < INPUT.machines_count; m++) {
        if (EA_THREADS[thread_id].population[solution_index].machine_compute_time[m] > worst_compute_time) {
            worst_compute_time_index = m;
            worst_compute_time = EA_THREADS[thread_id].population[solution_index].machine_compute_time[m];

        } else if (EA_THREADS[thread_id].population[solution_index].machine_compute_time[m] == worst_compute_time) {
            if (RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]) > 0.5) {
                worst_compute_time_index = m;
                worst_compute_time = EA_THREADS[thread_id].population[solution_index].machine_compute_time[m];
            }
        }
        
        if (EA_THREADS[thread_id].population[solution_index].machine_compute_time[m] < less_compute_time) {
            less_compute_time_index = m;
            less_compute_time = EA_THREADS[thread_id].population[solution_index].machine_compute_time[m];

        } else if (EA_THREADS[thread_id].population[solution_index].machine_compute_time[m] == less_compute_time) {
            if (RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]) > 0.5) {
                less_compute_time_index = m;
                less_compute_time = EA_THREADS[thread_id].population[solution_index].machine_compute_time[m];
            }
        }

        if (EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[m] > worst_energy) {
            worst_energy_index = m;
            worst_energy = EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[m];

        } else if (EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[m] == worst_energy) {
            if (RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]) > 0.5) {
                worst_energy_index = m;
                worst_energy = EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[m];
            }
        }
        
        if (EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[m] < less_energy) {
            less_energy_index = m;
            less_energy = EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[m];

        } else if (EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[m] == less_energy) {
            if (RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]) > 0.5) {
                less_energy_index = m;
                less_energy = EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[m];
            }
        }
    }

    worst_compute_time_machine_count = 0;
    worst_energy_machine_count = 0;
    less_compute_time_machine_count = 0;
    less_energy_machine_count = 0;

    int starting_offset = (int)(RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]) * INPUT.tasks_count);
    int current_task;

    /* Recolecto algunas tareas de las máquinas más representativas */
    for (int t = 0; (t < INPUT.tasks_count) &&
        ((worst_compute_time_machine_count < MAX__COLLECTED_TASKS) ||
        (worst_energy_machine_count < MAX__COLLECTED_TASKS) ||
        (less_compute_time_machine_count < MAX__COLLECTED_TASKS) ||
        (less_energy_machine_count < MAX__COLLECTED_TASKS)); t++) {

        current_task = starting_offset + t;
        if (current_task >= INPUT.tasks_count) {
            current_task = 0;
            starting_offset = -t;
        }

        if (worst_compute_time_machine_count < MAX__COLLECTED_TASKS) {
            if (EA_THREADS[thread_id].population[solution_index].task_assignment[current_task] == worst_compute_time_index) {
                worst_compute_time_machine_tasks[worst_compute_time_machine_count] = current_task;
                worst_compute_time_machine_count++;
            }
        }

        if (worst_energy_machine_count < MAX__COLLECTED_TASKS) {
            if (EA_THREADS[thread_id].population[solution_index].task_assignment[current_task] == worst_energy_index) {
                worst_energy_machine_tasks[worst_energy_machine_count] = current_task;
                worst_energy_machine_count++;
            }
        }
        
        if (less_compute_time_machine_count < MAX__COLLECTED_TASKS) {
            if (EA_THREADS[thread_id].population[solution_index].task_assignment[current_task] == less_compute_time_index) {
                less_compute_time_machine_tasks[less_compute_time_machine_count] = current_task;
                less_compute_time_machine_count++;
            }
        }

        if (less_energy_machine_count < MAX__COLLECTED_TASKS) {
            if (EA_THREADS[thread_id].population[solution_index].task_assignment[current_task] == less_energy_index) {
                less_energy_machine_tasks[less_energy_machine_count] = current_task;
                less_energy_machine_count++;
            }
        }
    }
    
    random  = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]);

    if (random < PALS__MAKESPAN_SEARCH_PROB) {
        search_type = PALS__MAKESPAN_SEARCH;
    } else if (random < PALS__MAKESPAN_SEARCH_PROB + PALS__ENERGY_SEARCH_PROB) {
        search_type = PALS__ENERGY_SEARCH;
    } else {
        search_type = PALS__MAKESPAN_SEARCH;
    }
    
    count_movements = 0;

    movements[thread_id].score = VERY_BIG_FLOAT;
    movements[thread_id].tipo = -1;
    movements[thread_id].src_task = -1;
    movements[thread_id].dst = -1;

    FLOAT machine_a_ct_old, machine_b_ct_old;
    FLOAT machine_a_ct_new, machine_b_ct_new;

    FLOAT machine_a_en_old, machine_b_en_old;
    FLOAT machine_a_en_new, machine_b_en_new;

    for (int i = 0; i < PALS__MAX_BUSQUEDAS; i++) {
        // Obtengo las tareas sorteadas.
        random1 = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]);
        if (random1 < PALS__RANDOM_SEARCH_PROB) {
            random1 = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]);
            task_x = (int)(random1 * INPUT.tasks_count);
            machine_a = EA_THREADS[thread_id].population[solution_index].task_assignment[task_x];
            
            assert(task_x >= 0);
            assert(task_x <= INPUT.tasks_count);
        } else if (search_type == PALS__MAKESPAN_SEARCH) {
            random1 = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]);
            task_x = worst_compute_time_machine_tasks[(int)(random1 * worst_compute_time_machine_count)];
            machine_a = worst_compute_time_index;

            assert(task_x >= 0);
            assert(task_x <= INPUT.tasks_count);
        } else if (search_type == PALS__ENERGY_SEARCH) {
            random1 = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]);
            task_x = worst_energy_machine_tasks[(int)(random1 * worst_energy_machine_count)];
            machine_a = worst_energy_index;
            
            assert(task_x >= 0);
            assert(task_x <= INPUT.tasks_count);
        }

        for (int loop = 0; loop < PALS__MAX_INTENTOS; loop++) {
            if (RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]) < PALS__SWAP_SEARCH) {
                movimiento = PALS__MOVIMIENTO_SWAP;
            } else {
                movimiento = PALS__MOVIMIENTO_MOVE;
            }

            score = 0.0;

            if (movimiento == PALS__MOVIMIENTO_SWAP) {
                // =================
                // Swap
                int task_y = 0;
                int machine_b = 0;

                random2 = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]);
                if (RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]) < 0.8) {
                    random2 = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]);
                    task_y = (int)(random2 * INPUT.tasks_count);
                    machine_b = EA_THREADS[thread_id].population[solution_index].task_assignment[task_y]; // Máquina b.

                    for (int i = 0; (i < INPUT.tasks_count) && (machine_a == machine_b); i++) {
                        task_y++;
                        if (task_y == INPUT.tasks_count) task_y = 0;

                        machine_b = EA_THREADS[thread_id].population[solution_index].task_assignment[task_y]; // Máquina b.
                    }
                    
                    assert(task_y >= 0);
                    assert(task_y <= INPUT.tasks_count);
                } else {
                    random2 = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]);
                    
                    if (search_type == PALS__MAKESPAN_SEARCH) {
                        if (less_compute_time_machine_count > 0) {
                            assert(less_compute_time_machine_count > 0);
                            
                            task_y = less_compute_time_machine_tasks[(int)(random2 * less_compute_time_machine_count)];
                            machine_b = less_compute_time_index;
                        } else {
                            task_y = (int)(random2 * INPUT.tasks_count);
                            machine_b = EA_THREADS[thread_id].population[solution_index].task_assignment[task_y]; // Máquina b.

                            for (int i = 0; (i < INPUT.tasks_count) && (machine_a == machine_b); i++) {
                                task_y++;
                                if (task_y == INPUT.tasks_count) task_y = 0;

                                machine_b = EA_THREADS[thread_id].population[solution_index].task_assignment[task_y]; // Máquina b.
                            }
                        }
                        
                        assert(task_y >= 0);
                        assert(task_y <= INPUT.tasks_count);
                    } else if (search_type == PALS__ENERGY_SEARCH) {
                        if (less_energy_machine_count > 0) {
                            assert(less_energy_machine_count > 0);
                            
                            task_y = less_energy_machine_tasks[(int)(random2 * less_energy_machine_count)];
                            machine_b = less_energy_index;
                        } else {
                            task_y = (int)(random2 * INPUT.tasks_count);
                            machine_b = EA_THREADS[thread_id].population[solution_index].task_assignment[task_y]; // Máquina b.

                            for (int i = 0; (i < INPUT.tasks_count) && (machine_a == machine_b); i++) {
                                task_y++;
                                if (task_y == INPUT.tasks_count) task_y = 0;

                                machine_b = EA_THREADS[thread_id].population[solution_index].task_assignment[task_y]; // Máquina b.
                            }
                        }
                        
                        assert(task_y >= 0);
                        assert(task_y <= INPUT.tasks_count);
                    }
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

                machine_a_en_old = EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[machine_a];
                machine_b_en_old = EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[machine_b];

                machine_a_en_new = machine_a_ct_new * get_scenario_energy_max(machine_a);
                machine_b_en_new = machine_b_ct_new * get_scenario_energy_max(machine_b);

                FLOAT max_old;
                max_old = machine_a_ct_old;
                if (max_old < machine_b_ct_old) max_old = machine_b_ct_old;

                score = compute_movement_score(thread_id, search_type,
                    worst_compute_time, worst_energy, 
                    machine_a_ct_new, machine_a_ct_old,
                    machine_b_ct_new, machine_b_ct_old,
                    machine_a_en_new, machine_a_en_old,
                    machine_b_en_new, machine_b_en_old);

                if (movements[thread_id].score > score) {
                    movements[thread_id].score = score;
                    movements[thread_id].tipo = PALS__MOVIMIENTO_SWAP;
                    movements[thread_id].src_task = task_x;
                    movements[thread_id].dst = task_y;
                }
            } else if (movimiento == PALS__MOVIMIENTO_MOVE) {
                // =================
                // Move
                int machine_b = 0;
                score = 0.0;

                // ================= Obtengo la tarea sorteada
                random2 = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]);
                if (RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]) < 0.8) {
                    random2 = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]);
                    machine_b = (int)(random2 * INPUT.machines_count);

                    for (int i = 0; (i < INPUT.machines_count) && (machine_a == machine_b); i++) {
                        machine_b++;
                        if (machine_b == INPUT.machines_count) machine_b = 0;
                    }
                } else {
                    if (search_type == PALS__MAKESPAN_SEARCH) {
                        machine_b = less_compute_time_index;
                    } else if (search_type == PALS__ENERGY_SEARCH) {
                        machine_b = less_energy_index;
                    }
                }

                machine_a_ct_old = EA_THREADS[thread_id].population[solution_index].machine_compute_time[machine_a];
                machine_b_ct_old = EA_THREADS[thread_id].population[solution_index].machine_compute_time[machine_b];

                // Calculo el score del swap sorteado.
                machine_a_ct_new = machine_a_ct_old - get_etc_value(machine_a, task_x); // Resto del ETC de x en a.
                machine_b_ct_new = machine_b_ct_old + get_etc_value(machine_b, task_x); // Sumo el ETC de x en b.

                machine_a_en_old = EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[machine_a];
                machine_b_en_old = EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[machine_b];

                machine_a_en_new = machine_a_ct_new * get_scenario_energy_max(machine_a);
                machine_b_en_new = machine_b_ct_new * get_scenario_energy_max(machine_b);

                score = compute_movement_score(thread_id, search_type,
                    worst_compute_time, worst_energy, 
                    machine_a_ct_new, machine_a_ct_old,
                    machine_b_ct_new, machine_b_ct_old,
                    machine_a_en_new, machine_a_en_old,
                    machine_b_en_new, machine_b_en_old);

                if (movements[thread_id].score > score) {
                    movements[thread_id].score = score;
                    movements[thread_id].tipo = PALS__MOVIMIENTO_MOVE;
                    movements[thread_id].src_task = task_x;
                    movements[thread_id].dst = machine_b;
                }
            }
        }
    }

    if (movements[thread_id].score < 0) {
        count_movements++;

        int task_src;
        int task_dst;
        int machine_src;
        int machine_dst;

        task_src = movements[thread_id].src_task;
        machine_src = EA_THREADS[thread_id].population[solution_index].task_assignment[task_src];

        if (movements[thread_id].tipo == PALS__MOVIMIENTO_SWAP) {
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

        } else if (movements[thread_id].tipo == PALS__MOVIMIENTO_MOVE) {
            machine_dst = movements[thread_id].dst;

            /* Hago el move */
            EA_THREADS[thread_id].population[solution_index].task_assignment[task_src] = machine_dst;

            EA_THREADS[thread_id].population[solution_index].machine_compute_time[machine_src] -= get_etc_value(machine_src, task_src);
            EA_THREADS[thread_id].population[solution_index].machine_compute_time[machine_dst] += get_etc_value(machine_dst, task_src);

            EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[machine_src] -=
                get_scenario_energy_max(machine_src) * get_etc_value(machine_src, task_src);
            EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[machine_dst] +=
                get_scenario_energy_max(machine_dst) * get_etc_value(machine_dst, task_src);

        }
        
        recompute_metrics(&EA_THREADS[thread_id].population[solution_index]);
        COUNT_EVALUATIONS[thread_id]++;
    }

    if (count_movements > 0) {
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

        FLOAT fitness_pre = fitness_zn(
            thread_id, makespan_pre, energy_pre,
            EA_THREADS[thread_id].makespan_nadir_value, EA_THREADS[thread_id].makespan_zenith_value,
            EA_THREADS[thread_id].energy_nadir_value, EA_THREADS[thread_id].energy_zenith_value);
                
        #ifdef DEBUG_1
            if (fitness_post < fitness_pre) {
                CHC_PALS_COUNT_FITNESS_IMPROV[thread_id]++;
        
                if (thread_id == 2) {
                    fprintf(stderr, "[DEBUG] PALS improv. %f -> %f, makespan(%.2f -> %.2f), energy(%.2f -> %.2f)\n",
                        fitness_pre, fitness_post, makespan_pre, EA_THREADS[thread_id].population[solution_index].makespan, 
                        energy_pre, EA_THREADS[thread_id].population[solution_index].energy_consumption);
                }
        
                if (search_type == PALS__MAKESPAN_SEARCH) {
                    CHC_PALS_COUNT_FITNESS_IMPROV_MAKESPAN[thread_id]++;
                } else if (search_type == PALS__ENERGY_SEARCH) {
                    CHC_PALS_COUNT_FITNESS_IMPROV_ENERGY[thread_id]++;
                }

                if (movements[thread_id].tipo == PALS__MOVIMIENTO_SWAP) {
                    CHC_PALS_COUNT_FITNESS_IMPROV_SWAP[thread_id]++;
                } else if (movements[thread_id].tipo == PALS__MOVIMIENTO_MOVE) {
                    CHC_PALS_COUNT_FITNESS_IMPROV_MOVE[thread_id]++;
                }
            }
            if (fitness_post > fitness_pre) {
                CHC_PALS_COUNT_FITNESS_DECLINE[thread_id]++;
            }
        #endif
        
        if (fitness_post < fitness_pre) return 1;
        else return 0;
    } else {
        return 0;
    }
}
