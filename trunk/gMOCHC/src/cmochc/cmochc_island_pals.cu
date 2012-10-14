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

#define PALS_MOVIMIENTO_SWAP 0
#define PALS_MOVIMIENTO_MOVE 1

#define MAX_COLLECTED_TASKS 16
//#define MAX_COLLECTED_TASKS 64

#define PALS_RANDOM_SEARCH 0
#define PALS_MAKESPAN_SEARCH 1
#define PALS_ENERGY_SEARCH 2

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

inline FLOAT compute_movement_score(int thread_id,
    FLOAT makespan, FLOAT energy_machine, 
    FLOAT machine_a_ct_new, FLOAT machine_a_ct_old,
    FLOAT machine_b_ct_new, FLOAT machine_b_ct_old,
    FLOAT machine_a_en_new, FLOAT machine_a_en_old,
    FLOAT machine_b_en_new, FLOAT machine_b_en_old) {
        
    FLOAT score = 0;
        
    #if defined(PALS__SIMPLE_FITNESS_0)
        if ((machine_a_ct_new > makespan) || (machine_b_ct_new > makespan) ||
            (machine_a_en_new > energy_machine) || (machine_b_en_new > energy_machine)) {

            score = VERY_BIG_FLOAT;
        } else {
            score = ((machine_a_ct_new - machine_a_ct_old) / makespan +
                    (machine_b_ct_new - machine_b_ct_old) / makespan) * EA_THREADS[thread_id].weight_makespan;

            score += ((machine_a_en_new - machine_a_en_old) / energy_machine +
                    (machine_b_en_new - machine_b_en_old) / energy_machine) * EA_THREADS[thread_id].weight_energy;
        }
    #endif
    #if defined(PALS__SIMPLE_FITNESS_1)
        if ((machine_a_ct_new > makespan) || (machine_b_ct_new > makespan) ||
            (machine_a_en_new - machine_a_en_old + machine_b_en_new - machine_b_en_old > 0)) {

            score = VERY_BIG_FLOAT;
        } else {
            score = (((machine_a_ct_new / machine_a_ct_old) - 1) +
                    ((machine_b_ct_new / machine_b_ct_old) - 1)) * EA_THREADS[thread_id].weight_makespan;

            score += (((machine_a_en_new / machine_a_en_old) - 1) +
                    ((machine_b_en_new / machine_b_en_old) - 1)) * EA_THREADS[thread_id].weight_energy;
        }
    #endif
    #if defined(PALS__SIMPLE_FITNESS_2)
        if (search_type == PALS_MAKESPAN_SEARCH) {
            if ((machine_a_ct_new > makespan) || (machine_b_ct_new > makespan)) {
                score = VERY_BIG_FLOAT;
            } else {
                score = ((machine_a_ct_new - machine_a_ct_old) / makespan +
                        (machine_b_ct_new - machine_b_ct_old) / makespan);
            }
        } else if (search_type == PALS_ENERGY_SEARCH) {
            if ((machine_a_en_new > energy_machine) || (machine_b_en_new > energy_machine)) {
                score = VERY_BIG_FLOAT;
            } else {
                score = ((machine_a_en_new - machine_a_en_old) / energy_machine +
                        (machine_b_en_new - machine_b_en_old) / energy_machine);
            }
        } else {
            if ((machine_a_ct_new > makespan) || (machine_b_ct_new > makespan) ||
                (machine_a_en_new > energy_machine) || (machine_b_en_new > energy_machine)) {
                score = VERY_BIG_FLOAT;
            } else {
                score = ((machine_a_ct_new - machine_a_ct_old) / makespan +
                        (machine_b_ct_new - machine_b_ct_old) / makespan) * EA_THREADS[thread_id].weight_makespan;

                score += ((machine_a_en_new - machine_a_en_old) / energy_machine +
                        (machine_b_en_new - machine_b_en_old) / energy_machine) * EA_THREADS[thread_id].weight_energy;
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
    
    return score;
}

void pals_search(int thread_id, int solution_index) {
    int worst_compute_time_machine_tasks[MAX_COLLECTED_TASKS];
    int worst_compute_time_machine_count;

    int worst_energy_machine_tasks[MAX_COLLECTED_TASKS];
    int worst_energy_machine_count;

    #ifdef DEBUG_1
        FLOAT makespan_pre = EA_THREADS[thread_id].population[solution_index].makespan;
        FLOAT energy_pre = EA_THREADS[thread_id].population[solution_index].energy_consumption;

        CHC_PALS_COUNT_EXECUTIONS[thread_id]++;
    #endif

    // Busco la máquina con mayor compute time.
    FLOAT makespan = EA_THREADS[thread_id].population[solution_index].machine_compute_time[0];
    int makespan_machine_index = 0;

    // Busco la máquina con mayor consumo de energy.
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

    worst_compute_time_machine_count = 0;
    worst_energy_machine_count = 0;

    int starting_offset = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]) * INPUT.tasks_count;
    int current_task;

    for (int t = 0; (t < INPUT.tasks_count) &&
        ((worst_compute_time_machine_count < MAX_COLLECTED_TASKS) ||
        (worst_energy_machine_count < MAX_COLLECTED_TASKS)); t++) {

        current_task = starting_offset + t;
        if (current_task >= INPUT.tasks_count) {
            current_task = 0;
            starting_offset = -t;
        }

        if (worst_compute_time_machine_count < MAX_COLLECTED_TASKS) {
            if (EA_THREADS[thread_id].population[solution_index].task_assignment[current_task] == makespan_machine_index) {
                worst_compute_time_machine_tasks[worst_compute_time_machine_count] = current_task;
                worst_compute_time_machine_count++;
            }
        }

        if (worst_energy_machine_count < MAX_COLLECTED_TASKS) {
            if (EA_THREADS[thread_id].population[solution_index].task_assignment[current_task] == energy_machine_index) {
                worst_energy_machine_tasks[worst_energy_machine_count] = current_task;
                worst_energy_machine_count++;
            }
        }
    }

    assert(worst_compute_time_machine_count > 0);
    assert(worst_energy_machine_count > 0);

    int search_type;
    FLOAT random;

    FLOAT score;
    int movimiento;

    FLOAT random1, random2;
    random  = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]);

    if (random < 0.05) {
        search_type = PALS_RANDOM_SEARCH;
    } else if (random < 0.90) {
        search_type = PALS_MAKESPAN_SEARCH;
    } else {
        search_type = PALS_ENERGY_SEARCH;
    }

    int count_movements = 0;
    int task_x;
    int machine_a;

    for (int i = 0; i < PALS__MAX_BUSQUEDAS; i++) {
        movements[thread_id].score = VERY_BIG_FLOAT;
        movements[thread_id].tipo = -1;
        movements[thread_id].src_task = -1;
        movements[thread_id].dst = -1;

        // Obtengo las tareas sorteadas.
        FLOAT machine_a_ct_old, machine_b_ct_old;
        FLOAT machine_a_ct_new, machine_b_ct_new;

        FLOAT machine_a_en_old, machine_b_en_old;
        FLOAT machine_a_en_new, machine_b_en_new;

        random1 = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]);

        if (search_type == PALS_RANDOM_SEARCH) {
            task_x = (int)(random1 * INPUT.tasks_count);
            machine_a = EA_THREADS[thread_id].population[solution_index].task_assignment[task_x]; // Máquina a.
        } else if (search_type == PALS_MAKESPAN_SEARCH) {
            task_x = worst_compute_time_machine_tasks[(int)(random1 * worst_compute_time_machine_count)];
            machine_a = makespan_machine_index;
        } else {
            task_x = worst_energy_machine_tasks[(int)(random1 * worst_energy_machine_count)];
            machine_a = energy_machine_index;
        }

        for (int loop = 0; loop < PALS__MAX_INTENTOS; loop++) {
            if (RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]) < PALS__SWAP_SEARCH) {
                movimiento = PALS_MOVIMIENTO_SWAP;
            } else {
                movimiento = PALS_MOVIMIENTO_MOVE;
            }

            score = 0.0;

            if (movimiento == PALS_MOVIMIENTO_SWAP) {
                // =================
                // Swap
                int task_y;
                int machine_b;

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

                machine_a_en_old = EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[machine_a];
                machine_b_en_old = EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[machine_b];

                machine_a_en_new = machine_b_ct_new * get_scenario_energy_max(machine_a);
                machine_b_en_new = machine_b_ct_new * get_scenario_energy_max(machine_b);

                FLOAT max_old;
                max_old = machine_a_ct_old;
                if (max_old < machine_b_ct_old) max_old = machine_b_ct_old;

                score = compute_movement_score(thread_id,
                    makespan, energy_machine, 
                    machine_a_ct_new, machine_a_ct_old,
                    machine_b_ct_new, machine_b_ct_old,
                    machine_a_en_new, machine_a_en_old,
                    machine_b_en_new, machine_b_en_old);

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

                machine_a_en_old = EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[machine_a];
                machine_b_en_old = EA_THREADS[thread_id].population[solution_index].machine_active_energy_consumption[machine_b];

                machine_a_en_new = machine_b_ct_new * get_scenario_energy_max(machine_a);
                machine_b_en_new = machine_b_ct_new * get_scenario_energy_max(machine_b);

                score = compute_movement_score(thread_id,
                    makespan, energy_machine, 
                    machine_a_ct_new, machine_a_ct_old,
                    machine_b_ct_new, machine_b_ct_old,
                    machine_a_en_new, machine_a_en_old,
                    machine_b_en_new, machine_b_en_old);

                if (movements[thread_id].score > score) {
                    movements[thread_id].score = score;
                    movements[thread_id].tipo = PALS_MOVIMIENTO_MOVE;
                    movements[thread_id].src_task = task_x;
                    movements[thread_id].dst = machine_b;
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

    if (count_movements > 0) {
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
    }

    #ifdef DEBUG_1
        FLOAT fitness_post = fitness(thread_id, solution_index);

        FLOAT fitness_pre = fitness_zn(
            thread_id, makespan_pre, energy_pre,
            EA_THREADS[thread_id].makespan_nadir_value, EA_THREADS[thread_id].makespan_zenith_value,
            EA_THREADS[thread_id].energy_nadir_value, EA_THREADS[thread_id].energy_zenith_value);
        /*
        #ifdef DEBUG_3
            fprintf(stderr, "[DEBUG] <thread=%d> PALS fitness from %.4f to %.4f (makespan %.2f->%.2f / energy %.2f->%.2f)\n",
                thread_id, fitness_pre, fitness_post, makespan_pre, EA_THREADS[thread_id].population[solution_index].makespan,
                energy_pre, EA_THREADS[thread_id].population[solution_index].energy_consumption);
        #endif
        */
        if (fitness_post < fitness_pre) {
            CHC_PALS_COUNT_FITNESS_IMPROV[thread_id]++;
    
            if (search_type == PALS_RANDOM_SEARCH) {
                CHC_PALS_COUNT_FITNESS_IMPROV_RANDOM[thread_id]++;
            } else if (search_type == PALS_MAKESPAN_SEARCH) {
                CHC_PALS_COUNT_FITNESS_IMPROV_MAKESPAN[thread_id]++;
            } else {
                CHC_PALS_COUNT_FITNESS_IMPROV_ENERGY[thread_id]++;
            }
        }
        if (fitness_post > fitness_pre) {
            CHC_PALS_COUNT_FITNESS_DECLINE[thread_id]++;
        }
    #endif
}
