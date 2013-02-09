#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <assert.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <semaphore.h>

#include "../config.h"
#include "../utils.h"
#include "../basic/mct.h"
#include "../basic/minmin.h"
#include "../basic/pminmin.h"
#include "../random/cpu_rand.h"
#include "../random/cpu_drand48.h"
#include "../random/cpu_mt.h"

#include "me_rpals_cpu.h"

void me_rpals_cpu(struct params &input, struct etc_matrix *etc, struct energy_matrix *energy)
{
    // ==============================================================================
    // rPALS
    // ==============================================================================

    timespec ts_total_time_end;
    clock_gettime(CLOCK_REALTIME, &ts_total_time_end);

    // Timming -----------------------------------------------------
    timespec ts_init;
    timming_start(ts_init);
    // Timming -----------------------------------------------------

    timespec ts_total_time_start;
    clock_gettime(CLOCK_REALTIME, &ts_total_time_start);

    // Inicializo la memoria y los hilos de ejecucin.
    struct rpals_cpu_instance instance;
    rpals_cpu_init(input, etc, energy, input.seed, instance);

    // Timming -----------------------------------------------------
    timming_end(">> rpals_cpu_init", ts_init);
    // Timming -----------------------------------------------------

    double random = 0.0;

    // RPALS_CPUK__INIT_POP ===================================================================
   
    init_empty_solution(etc, energy, &(instance.population[0]));
    init_empty_solution(etc, energy, &(instance.population[1]));

    #if defined(RPALS_INIT_PMINMIN)
        printf("[DEBUG] Usando pMin-Min\n");

        compute_pminmin(instance.etc,
            &(instance.population[0]),
            instance.count_threads);
    #endif
    #if defined(RPALS_INIT_MINMIN)
        printf("[DEBUG] Usando Min-Min\n");

        compute_minmin(&(instance.population[0]));
    #endif
    #if defined(RPALS_INIT_MCT)
        printf("[DEBUG] Usando MCT\n");

        compute_mct(&(instance.population[0]));
    #endif

    fprintf(stdout, " >> Starting makespan: %f // energy: %f\n",
            get_makespan(&instance.population[0]), get_energy(&instance.population[0]));

    #if defined(DEBUG_DEV)
        validate_solution(&(instance.population[0]));
    #endif
    
    instance.best_solution = 0;
    
    // ==================================================================
    {
        float original_makespan;
        float original_energy;
        int search_type;
        double search_type_random;
        int machine_a, machine_b;
        int machine_a_task_count;
        int machine_b_task_count;
        int task_x;
        float machine_a_energy_idle;
        float machine_a_energy_max;
        float machine_b_energy_idle;
        float machine_b_energy_max;
        float machine_a_ct_old, machine_b_ct_old;
        float machine_a_ct_new, machine_b_ct_new;
        float current_makespan;
        int task_x_pos;
        int task_x_current;
        int machine_b_current;
        int top_task_a;
        int top_task_b;
        int top_machine_b;
        int task_y;
        float best_delta_makespan;
        float best_delta_energy;
        int task_x_best_move_pos;
        int machine_b_best_move_id;
        int task_x_best_swap_pos;
        int task_y_best_swap_pos;
        int task_x_offset;
        int mov_type;
        int task_y_pos, task_y_current;
        int task_y_offset;
        float swap_diff_energy;
        int machine_b_offset;
        float machine_b_current_energy_idle;
        float machine_b_current_energy_max;

        timespec ts_current;
        clock_gettime(CLOCK_REALTIME, &ts_current);

        while ((ts_current.tv_sec - ts_total_time_end.tv_sec < instance.max_time_secs) &&
            (instance.total_iterations < instance.max_iterations))
        {
            clock_gettime(CLOCK_REALTIME, &ts_current);

            // RPALS_CPU_WORK__SEARCH ====================================================================
            struct solution *selected_solution;
            selected_solution = &(instance.population[(instance.best_solution + 1) % 2]);
            clone_solution(selected_solution, &(instance.population[0]), 0);

            // Determino la estrategia de busqueda del hilo  =====================================================
            #if defined(DEBUG_DEV)
                fprintf(stdout, "[DEBUG] Selected individual\n");
                fprintf(stdout, "        Original_solutiol_pos = %d\n", current_sol_pos);
                fprintf(stdout, "        Selected_solution_pos = %d\n", selected_solution_pos);
                fprintf(stdout, "        Selected_solution.status = %d\n", selected_solution->status);
                fprintf(stdout, "        Selected_solution.initializd = %d\n", selected_solution->initialized);
            #endif

            original_makespan = get_makespan(selected_solution);
            original_energy = get_energy(selected_solution);

            search_type_random = 0.0;

            #ifdef CPU_MERSENNE_TWISTER
                search_type_random = cpu_mt_generate(instance.random_state);
            #endif
            #ifdef CPU_RAND
                search_type_random = cpu_rand_generate(instance.random_state);
            #endif
            #ifdef CPU_DRAND48
                search_type_random = cpu_drand48_generate(instance.random_state);
            #endif

            if (search_type_random < RPALS_CPU_SEARCH_BALANCE__MAKESPAN)
            {
                search_type = RPALS_CPU_SEARCH__MAKESPAN_GREEDY;
            }
            else if (search_type_random < RPALS_CPU_SEARCH_BALANCE__MAKESPAN + RPALS_CPU_SEARCH_BALANCE__ENERGY)
            {
                search_type = RPALS_CPU_SEARCH__ENERGY_GREEDY;
            }
            else
            {
                search_type = RPALS_CPU_SEARCH__RANDOM_GREEDY;
            }

            #ifdef CPU_MERSENNE_TWISTER
                random = cpu_mt_generate(instance.random_state);
            #endif
            #ifdef CPU_RAND
                random = cpu_rand_generate(instance.random_state);
            #endif
            #ifdef CPU_DRAND48
                random = cpu_drand48_generate(instance.random_state);
            #endif

            /*work_iteration_size = (int)floor(RPALS_CPU_WORK__THREAD_ITERATIONS * random);

            for (search_iteration = 0; search_iteration < work_iteration_size; search_iteration++)
            {*/
                instance.total_iterations++;

                // Determino las maquinas de inicio para la busqueda.
                if (search_type == RPALS_CPU_SEARCH__MAKESPAN_GREEDY)
                {
                    //fprintf(stdout, "[DEBUG] Makespan greedy\n");

                    #ifdef CPU_MERSENNE_TWISTER
                        random = cpu_mt_generate(instance.random_state);
                    #endif
                    #ifdef CPU_RAND
                        random = cpu_rand_generate(instance.random_state);
                    #endif
                    #ifdef CPU_DRAND48
                        random = cpu_drand48_generate(instance.random_state);
                    #endif

                    if (random > RPALS_CPU_SEARCH__MAKESPAN_GREEDY_PSEL_WORST) {
                        #ifdef CPU_MERSENNE_TWISTER
                            random = cpu_mt_generate(instance.random_state);
                        #endif
                        #ifdef CPU_RAND
                            random = cpu_rand_generate(instance.random_state);
                        #endif
                        #ifdef CPU_DRAND48
                            random = cpu_drand48_generate(instance.random_state);
                        #endif

                        machine_a = (int)floor(random * etc->machines_count);
                    } else {
                        machine_a = get_worst_ct_machine_id(selected_solution);
                    }

                    #ifdef CPU_MERSENNE_TWISTER
                        random = cpu_mt_generate(instance.random_state);
                    #endif
                    #ifdef CPU_RAND
                        random = cpu_rand_generate(instance.random_state);
                    #endif
                    #ifdef CPU_DRAND48
                        random = cpu_drand48_generate(instance.random_state);
                    #endif

                    if (random > RPALS_CPU_SEARCH__MAKESPAN_GREEDY_PSEL_BEST) {
                        #ifdef CPU_MERSENNE_TWISTER
                            random = cpu_mt_generate(instance.random_state);
                        #endif
                        #ifdef CPU_RAND
                            random = cpu_rand_generate(instance.random_state);
                        #endif
                        #ifdef CPU_DRAND48
                            random = cpu_drand48_generate(instance.random_state);
                        #endif

                        // Siempre selecciono la segunda mquina aleatoriamente.
                        machine_b = (int)floor(random * (etc->machines_count - 1));

                        //fprintf(stdout, "[DEBUG] Random machine_b = %d\n", machine_b);
                    } else {
                        machine_b = get_best_ct_machine_id(selected_solution);

                        //fprintf(stdout, "[DEBUG] Best CT machine_b = %d\n", machine_b);
                    }
                }
                else if (search_type == RPALS_CPU_SEARCH__ENERGY_GREEDY)
                {
                    //fprintf(stdout, "[DEBUG] Energy greedy\n");

                    #ifdef CPU_MERSENNE_TWISTER
                        random = cpu_mt_generate(instance.random_state);
                    #endif
                    #ifdef CPU_RAND
                        random = cpu_rand_generate(instance.random_state);
                    #endif
                    #ifdef CPU_DRAND48
                        random = cpu_drand48_generate(instance.random_state);
                    #endif

                    if (random > RPALS_CPU_SEARCH__ENERGY_GREEDY_PSEL_WORST) {
                        #ifdef CPU_MERSENNE_TWISTER
                            random = cpu_mt_generate(instance.random_state);
                        #endif
                        #ifdef CPU_RAND
                            random = cpu_rand_generate(instance.random_state);
                        #endif
                        #ifdef CPU_DRAND48
                            random = cpu_drand48_generate(instance.random_state);
                        #endif

                        machine_a = (int)floor(random * etc->machines_count);

                        //fprintf(stdout, "[DEBUG] Random machine_a = %d\n", machine_a);
                    } else {
                        machine_a = get_worst_energy_machine_id(selected_solution);

                        //fprintf(stdout, "[DEBUG] Worst energy machine_a = %d\n", machine_a);
                    }

                    #ifdef CPU_MERSENNE_TWISTER
                        random = cpu_mt_generate(instance.random_state);
                    #endif
                    #ifdef CPU_RAND
                        random = cpu_rand_generate(instance.random_state);
                    #endif
                    #ifdef CPU_DRAND48
                        random = cpu_drand48_generate(instance.random_state);
                    #endif

                    if (random > RPALS_CPU_SEARCH__ENERGY_GREEDY_PSEL_BEST) {
                        #ifdef CPU_MERSENNE_TWISTER
                            random = cpu_mt_generate(instance.random_state);
                        #endif
                        #ifdef CPU_RAND
                            random = cpu_rand_generate(instance.random_state);
                        #endif
                        #ifdef CPU_DRAND48
                            random = cpu_drand48_generate(instance.random_state);
                        #endif

                        // Siempre selecciono la segunda mquina aleatoriamente.
                        machine_b = (int)floor(random * (etc->machines_count - 1));

                        //fprintf(stdout, "[DEBUG] Random machine_b = %d\n", machine_b);
                    } else {
                        machine_b = get_best_energy_machine_id(selected_solution);

                        //fprintf(stdout, "[DEBUG] Worst energy machine_b = %d\n", machine_b);
                    }
                }
                else
                {
                    #ifdef CPU_MERSENNE_TWISTER
                        random = cpu_mt_generate(instance.random_state);
                    #endif
                    #ifdef CPU_RAND
                        random = cpu_rand_generate(instance.random_state);
                    #endif
                    #ifdef CPU_DRAND48
                        random = cpu_drand48_generate(instance.random_state);
                    #endif

                    // La estrategia es aleatoria.
                    machine_a = (int)floor(random * etc->machines_count);

                    #ifdef CPU_MERSENNE_TWISTER
                        random = cpu_mt_generate(instance.random_state);
                    #endif
                    #ifdef CPU_RAND
                        random = cpu_rand_generate(instance.random_state);
                    #endif
                    #ifdef CPU_DRAND48
                        random = cpu_drand48_generate(instance.random_state);
                    #endif

                    // Siempre selecciono la segunda mquina aleatoriamente.
                    machine_b = (int)floor(random * (etc->machines_count - 1));
                }

                machine_a_task_count = get_machine_tasks_count(selected_solution, machine_a);
                while (machine_a_task_count == 0) {
                    machine_a = (machine_a + 1) % etc->machines_count;
                    machine_a_task_count = get_machine_tasks_count(selected_solution, machine_a);
                }

                if (machine_a == machine_b) machine_b = (machine_b + 1) % etc->machines_count;
                machine_b_task_count = get_machine_tasks_count(selected_solution, machine_b);

                #ifdef CPU_MERSENNE_TWISTER
                    random = cpu_mt_generate(instance.random_state);
                #endif
                #ifdef CPU_RAND
                    random = cpu_rand_generate(instance.random_state);
                #endif
                #ifdef CPU_DRAND48
                    random = cpu_drand48_generate(instance.random_state);
                #endif

                task_x = (int)floor(random * machine_a_task_count);

                machine_a_energy_idle = get_energy_idle_value(energy, machine_a);
                machine_a_energy_max = get_energy_max_value(energy, machine_a);
                machine_b_energy_idle = get_energy_idle_value(energy, machine_b);
                machine_b_energy_max = get_energy_max_value(energy, machine_b);

                current_makespan = get_makespan(selected_solution);

                #ifdef CPU_MERSENNE_TWISTER
                    random = cpu_mt_generate(instance.random_state);
                #endif
                #ifdef CPU_RAND
                    random = cpu_rand_generate(instance.random_state);
                #endif
                #ifdef CPU_DRAND48
                    random = cpu_drand48_generate(instance.random_state);
                #endif

                top_task_a = (int)floor(random * RPALS_CPU_WORK__SRC_TASK_NHOOD) + 1;
                if (top_task_a > machine_a_task_count) top_task_a = machine_a_task_count;

                #ifdef CPU_MERSENNE_TWISTER
                    random = cpu_mt_generate(instance.random_state);
                #endif
                #ifdef CPU_RAND
                    random = cpu_rand_generate(instance.random_state);
                #endif
                #ifdef CPU_DRAND48
                    random = cpu_drand48_generate(instance.random_state);
                #endif

                top_task_b = (int)floor(random * RPALS_CPU_WORK__DST_TASK_NHOOD) + 1;
                if (top_task_b > machine_b_task_count) top_task_b = machine_b_task_count;

                #ifdef CPU_MERSENNE_TWISTER
                    random = cpu_mt_generate(instance.random_state);
                #endif
                #ifdef CPU_RAND
                    random = cpu_rand_generate(instance.random_state);
                #endif
                #ifdef CPU_DRAND48
                    random = cpu_drand48_generate(instance.random_state);
                #endif

                top_machine_b = (int)floor(random * RPALS_CPU_WORK__DST_MACH_NHOOD) + 1;
                if (top_machine_b > etc->machines_count) top_machine_b = etc->machines_count;

                #ifdef CPU_MERSENNE_TWISTER
                    random = cpu_mt_generate(instance.random_state);
                #endif
                #ifdef CPU_RAND
                    random = cpu_rand_generate(instance.random_state);
                #endif
                #ifdef CPU_DRAND48
                    random = cpu_drand48_generate(instance.random_state);
                #endif

                task_y = (int)floor(random * machine_b_task_count);

                best_delta_makespan = current_makespan;
                best_delta_energy = 0.0;
                task_x_best_move_pos = -1;
                machine_b_best_move_id = -1;
                task_x_best_swap_pos = -1;
                task_y_best_swap_pos = -1;

                for (task_x_offset = 0; (task_x_offset < top_task_a); task_x_offset++)
                {
                    task_x_pos = (task_x + task_x_offset) % machine_a_task_count;
                    task_x_current = get_machine_task_id(selected_solution, machine_a, task_x_pos);

                    // Determino que tipo movimiento va a realizar el hilo.
                    #ifdef CPU_MERSENNE_TWISTER
                        random = cpu_mt_generate(instance.random_state);
                    #endif
                    #ifdef CPU_RAND
                        random = cpu_rand_generate(instance.random_state);
                    #endif
                    #ifdef CPU_DRAND48
                        random = cpu_drand48_generate(instance.random_state);
                    #endif

                    mov_type = RPALS_CPU_SEARCH_OP__SWAP;
                    if ((random < RPALS_CPU_SEARCH_OP_BALANCE__SWAP) && (machine_b_task_count > 0))
                    {
                        mov_type = RPALS_CPU_SEARCH_OP__SWAP;
                    }
                    else //if (random < RPALS_CPU_SEARCH_OP_BALANCE__SWAP + RPALS_CPU_SEARCH_OP_BALANCE__MOVE)
                    {
                        mov_type = RPALS_CPU_SEARCH_OP__MOVE;
                    }

                    if (mov_type == RPALS_CPU_SEARCH_OP__SWAP)
                    {
                        for (task_y_offset = 0; (task_y_offset < top_task_b); task_y_offset++)
                        {
                            task_y_pos = (task_y + task_y_offset) % machine_b_task_count;
                            task_y_current = get_machine_task_id(selected_solution, machine_b, task_y_pos);

                            // Mquina 1.
                            machine_a_ct_old = get_machine_compute_time(selected_solution, machine_a);

                            machine_a_ct_new = machine_a_ct_old -
                                get_etc_value(etc, machine_a, task_x_current) +
                                get_etc_value(etc, machine_a, task_y_current);

                            // Mquina 2.
                            machine_b_ct_old = get_machine_compute_time(selected_solution, machine_b);

                            machine_b_ct_new = machine_b_ct_old -
                                get_etc_value(etc, machine_b, task_y_current) +
                                get_etc_value(etc, machine_b, task_x_current);

                            #ifdef CPU_MERSENNE_TWISTER
                                random = cpu_mt_generate(instance.random_state);
                            #endif
                            #ifdef CPU_RAND
                                random = cpu_rand_generate(instance.random_state);
                            #endif
                            #ifdef CPU_DRAND48
                                random = cpu_drand48_generate(instance.random_state);
                            #endif

                            if ((search_type == RPALS_CPU_SEARCH__MAKESPAN_GREEDY) ||
                                ((random < 0.5) && (search_type == RPALS_CPU_SEARCH__RANDOM_GREEDY)))
                            {
                                swap_diff_energy =
                                    ((machine_a_ct_old - machine_a_ct_new) * (machine_a_energy_max - machine_a_energy_idle)) +
                                    ((machine_b_ct_old - machine_b_ct_new) * (machine_b_energy_max - machine_b_energy_idle));

                                if (machine_b_ct_new <= machine_a_ct_new)
                                {
                                    if (machine_a_ct_new < best_delta_makespan) {
                                        best_delta_makespan = machine_a_ct_new;
                                        best_delta_energy = swap_diff_energy;
                                        task_x_best_swap_pos = task_x_pos;
                                        task_y_best_swap_pos = task_y_pos;
                                        task_x_best_move_pos = -1;
                                        machine_b_best_move_id = -1;
                                    } else if (floor(machine_a_ct_new) == floor(best_delta_makespan)) {
                                        if (swap_diff_energy > best_delta_energy)
                                        {
                                            best_delta_energy = swap_diff_energy;
                                            best_delta_makespan = machine_a_ct_new;
                                            task_x_best_swap_pos = task_x_pos;
                                            task_y_best_swap_pos = task_y_pos;
                                            task_x_best_move_pos = -1;
                                            machine_b_best_move_id = -1;
                                        }
                                    }
                                }
                                else if (machine_a_ct_new <= machine_b_ct_new)
                                {
                                    if (machine_b_ct_new < best_delta_makespan) {
                                        best_delta_makespan = machine_b_ct_new;
                                        best_delta_energy = swap_diff_energy;
                                        task_x_best_swap_pos = task_x_pos;
                                        task_y_best_swap_pos = task_y_pos;
                                        task_x_best_move_pos = -1;
                                        machine_b_best_move_id = -1;
                                    } else if (floor(machine_b_ct_new) == floor(best_delta_makespan)) {
                                        if (swap_diff_energy > best_delta_energy)
                                        {
                                            best_delta_energy = swap_diff_energy;
                                            best_delta_makespan = machine_b_ct_new;
                                            task_x_best_swap_pos = task_x_pos;
                                            task_y_best_swap_pos = task_y_pos;
                                            task_x_best_move_pos = -1;
                                            machine_b_best_move_id = -1;
                                        }
                                    }
                                }
                            }

                            if ((search_type == RPALS_CPU_SEARCH__ENERGY_GREEDY) ||
                                ((random >= 0.5) && (search_type == RPALS_CPU_SEARCH__RANDOM_GREEDY)))
                            {
                                swap_diff_energy =
                                    ((machine_a_ct_old - machine_a_ct_new) * (machine_a_energy_max - machine_a_energy_idle)) +
                                    ((machine_b_ct_old - machine_b_ct_new) * (machine_b_energy_max - machine_b_energy_idle));

                                if ((swap_diff_energy > best_delta_energy) &&
                                    (machine_a_ct_new <= current_makespan) &&
                                    (machine_b_ct_new <= current_makespan))
                                {
                                    best_delta_energy = swap_diff_energy;

                                    if (machine_a_ct_new <= machine_b_ct_new) best_delta_makespan = machine_b_ct_new;
                                    else best_delta_makespan = machine_a_ct_new;

                                    task_x_best_swap_pos = task_x_pos;
                                    task_y_best_swap_pos = task_y_pos;
                                    task_x_best_move_pos = -1;
                                    machine_b_best_move_id = -1;
                                } else if (floor(swap_diff_energy) == floor(best_delta_energy)) {
                                    if ((machine_b_ct_new <= machine_a_ct_new) && (machine_a_ct_new < best_delta_makespan))
                                    {
                                        best_delta_makespan = machine_a_ct_new;
                                        best_delta_energy = swap_diff_energy;
                                        task_x_best_swap_pos = task_x_pos;
                                        task_y_best_swap_pos = task_y_pos;
                                        task_x_best_move_pos = -1;
                                        machine_b_best_move_id = -1;
                                    }
                                    else if ((machine_a_ct_new <= machine_b_ct_new) && (machine_b_ct_new < best_delta_makespan))
                                    {
                                        best_delta_makespan = machine_b_ct_new;
                                        best_delta_energy = swap_diff_energy;
                                        task_x_best_swap_pos = task_x_pos;
                                        task_y_best_swap_pos = task_y_pos;
                                        task_x_best_move_pos = -1;
                                        machine_b_best_move_id = -1;
                                    }
                                }
                            }
                        }    // Termino el loop de TASK_B
                    }
                    else if (mov_type == RPALS_CPU_SEARCH_OP__MOVE)
                    {
                        machine_b_current = machine_b;

                        for (machine_b_offset = 0; (machine_b_offset < top_machine_b); machine_b_offset++)
                        {
                            if (machine_b_offset == 1) {
                                #ifdef CPU_MERSENNE_TWISTER
                                    random = cpu_mt_generate(instance.random_state);
                                #endif
                                #ifdef CPU_RAND
                                    random = cpu_rand_generate(instance.random_state);
                                #endif
                                #ifdef CPU_DRAND48
                                    random = cpu_drand48_generate(instance.random_state);
                                #endif

                                // Siempre selecciono la segunda mquina aleatoriamente.
                                machine_b_current = (int)floor(random * (etc->machines_count - 1));

                                if (machine_b_current == machine_a) machine_b_current = (machine_b_current + 1) % etc->machines_count;
                            } 
                            else if (machine_b_offset > 1) {
                                if (machine_b + machine_b_offset != machine_a) {
                                    machine_b_current = (machine_b + machine_b_offset) % etc->machines_count;
                                } else {
                                    machine_b_current = (machine_b + machine_b_offset + 1) % etc->machines_count;
                                }
                            }

                            if (machine_b_current != machine_a)
                            {
                                machine_b_current_energy_idle = get_energy_idle_value(energy, machine_b_current);
                                machine_b_current_energy_max = get_energy_max_value(energy, machine_b_current);

                                // Mquina 1.
                                machine_a_ct_old = get_machine_compute_time(selected_solution, machine_a);
                                machine_a_ct_new = machine_a_ct_old - get_etc_value(etc, machine_a, task_x_current);

                                // Mquina 2.
                                machine_b_ct_old = get_machine_compute_time(selected_solution, machine_b_current);
                                machine_b_ct_new = machine_b_ct_old + get_etc_value(etc, machine_b_current, task_x_current);

                                #ifdef CPU_MERSENNE_TWISTER
                                    random = cpu_mt_generate(instance.random_state);
                                #endif
                                #ifdef CPU_RAND
                                    random = cpu_rand_generate(instance.random_state);
                                #endif
                                #ifdef CPU_DRAND48
                                    random = cpu_drand48_generate(instance.random_state);
                                #endif

                                if ((search_type == RPALS_CPU_SEARCH__MAKESPAN_GREEDY) ||
                                    ((random < 0.5) && (search_type == RPALS_CPU_SEARCH__RANDOM_GREEDY)))
                                {
                                    swap_diff_energy =
                                        ((machine_a_ct_old - machine_a_ct_new) * (machine_a_energy_max - machine_a_energy_idle)) +
                                        ((machine_b_ct_old - machine_b_ct_new) * (machine_b_current_energy_max - machine_b_current_energy_idle));

                                    if (machine_b_ct_new <= machine_a_ct_new)
                                    {
                                        if (machine_a_ct_new < best_delta_makespan) {
                                            best_delta_makespan = machine_a_ct_new;
                                            best_delta_energy = swap_diff_energy;
                                            task_x_best_swap_pos = -1;
                                            task_y_best_swap_pos = -1;
                                            task_x_best_move_pos = task_x_pos;
                                            machine_b_best_move_id = machine_b_current;
                                        } else if (floor(machine_a_ct_new) == floor(best_delta_makespan)) {
                                            if (swap_diff_energy > best_delta_energy)
                                            {
                                                best_delta_energy = swap_diff_energy;
                                                best_delta_makespan = machine_a_ct_new;
                                                task_x_best_swap_pos = -1;
                                                task_y_best_swap_pos = -1;
                                                task_x_best_move_pos = task_x_pos;
                                                machine_b_best_move_id = machine_b_current;
                                            }
                                        }
                                    }
                                    else if (machine_a_ct_new <= machine_b_ct_new)
                                    {
                                        if (machine_b_ct_new < best_delta_makespan) {
                                            best_delta_makespan = machine_b_ct_new;
                                            best_delta_energy = swap_diff_energy;
                                            task_x_best_swap_pos = -1;
                                            task_y_best_swap_pos = -1;
                                            task_x_best_move_pos = task_x_pos;
                                            machine_b_best_move_id = machine_b_current;
                                        } 
                                        else if (floor(machine_b_ct_new) == floor(best_delta_makespan)) {
                                            if (swap_diff_energy > best_delta_energy) {
                                                best_delta_energy = swap_diff_energy;
                                                best_delta_makespan = machine_b_ct_new;
                                                task_x_best_swap_pos = -1;
                                                task_y_best_swap_pos = -1;
                                                task_x_best_move_pos = task_x_pos;
                                                machine_b_best_move_id = machine_b_current;
                                            }
                                        }
                                    }
                                }

                                if ((search_type == RPALS_CPU_SEARCH__ENERGY_GREEDY) ||
                                    ((random >= 0.5) && (search_type == RPALS_CPU_SEARCH__RANDOM_GREEDY)))
                                {
                                    swap_diff_energy =
                                        ((machine_a_ct_old - machine_a_ct_new) * (machine_a_energy_max - machine_a_energy_idle)) +
                                        ((machine_b_ct_old - machine_b_ct_new) * (machine_b_current_energy_max - machine_b_current_energy_idle));

                                    if ((swap_diff_energy > best_delta_energy) &&
                                        (machine_a_ct_new <= current_makespan) &&
                                        (machine_b_ct_new <= current_makespan))
                                    {
                                        best_delta_energy = swap_diff_energy;

                                        if (machine_a_ct_new <= machine_b_ct_new) best_delta_makespan = machine_b_ct_new;
                                        else best_delta_makespan = machine_a_ct_new;

                                        task_x_best_swap_pos = -1;
                                        task_y_best_swap_pos = -1;
                                        task_x_best_move_pos = task_x_pos;
                                        machine_b_best_move_id = machine_b_current;
                                    } else if (floor(swap_diff_energy) == floor(best_delta_energy)) {
                                        if ((machine_b_ct_new <= machine_a_ct_new) && (machine_a_ct_new < best_delta_makespan))
                                        {
                                            best_delta_makespan = machine_a_ct_new;
                                            best_delta_energy = swap_diff_energy;
                                            task_x_best_swap_pos = -1;
                                            task_y_best_swap_pos = -1;
                                            task_x_best_move_pos = task_x_pos;
                                            machine_b_best_move_id = machine_b_current;
                                        }
                                        else if ((machine_a_ct_new <= machine_b_ct_new) && (machine_b_ct_new < best_delta_makespan))
                                        {
                                            best_delta_makespan = machine_b_ct_new;
                                            best_delta_energy = swap_diff_energy;
                                            task_x_best_swap_pos = -1;
                                            task_y_best_swap_pos = -1;
                                            task_x_best_move_pos = task_x_pos;
                                            machine_b_best_move_id = machine_b_current;
                                        }
                                    }
                                }
                            }
                        }    // Termino el loop de MACHINE_B
                    }        // Termino el IF de SWAP/MOVE
                }            // Termino el loop de TASK_A

                // Hago los cambios ======================================================================================
                if ((task_x_best_swap_pos != -1) && (task_y_best_swap_pos != -1))
                {
                    // Intercambio las tareas!
                    #if defined(DEBUG_DEV) 
                    fprintf(stdout, "[DEBUG] Ejecuto un SWAP! %f %f (%d, %d, %d, %d)\n",
                        best_delta_makespan, best_delta_energy, machine_a, task_x_best_swap_pos, machine_b, task_y_best_swap_pos);
                    #endif
                    
                    swap_tasks_by_pos(selected_solution, machine_a, task_x_best_swap_pos, machine_b, task_y_best_swap_pos);
                }
                else if ((task_x_best_move_pos != -1) && (machine_b_best_move_id != -1))
                {
                    // Muevo la tarea!
                    #if defined(DEBUG_DEV) 
                    fprintf(stdout, "[DEBUG] Ejecuto un MOVE! %f %f (%d, %d, %d)\n",
                        best_delta_makespan, best_delta_energy, machine_a, task_x_best_move_pos, machine_b_best_move_id);
                    #endif
                    
                    move_task_to_machine_by_pos(selected_solution, machine_a, task_x_best_move_pos, machine_b_best_move_id);
                }

                if ((original_makespan > get_makespan(selected_solution)) ||
                    ((original_makespan == get_makespan(selected_solution)) &&
                    (original_energy > get_energy(selected_solution)))) 
                {
                    instance.best_solution = (instance.best_solution + 1) % 2;
                }
            //}
        }
    }
    // ======================================================

    // Timming -----------------------------------------------------
    timespec ts_finalize;
    timming_start(ts_finalize);
    // Timming -----------------------------------------------------

    #if defined(DEBUG)
        fprintf(stdout, "%f %f (%d)\n", 
            get_makespan(&(instance.population[instance.best_solution])), 
            get_energy(&(instance.population[instance.best_solution])),
            instance.population[instance.best_solution].status);
    #else
    if (!OUTPUT_SOLUTION)
    {
        fprintf(stdout, "%f %f (%d)\n", 
            get_makespan(&(instance.population[instance.best_solution])), 
            get_energy(&(instance.population[instance.best_solution])),
            instance.population[instance.best_solution].status);
    }
    else
    {
        for (int task = 0; task < etc->tasks_count; task++)
        {
            fprintf(stdout, "%d\n", get_task_assigned_machine_id(&(instance.population[instance.best_solution]), task));
        }
    }
    #endif

    // Libero la memoria del dispositivo.
    rpals_cpu_finalize(instance);

    // Timming -----------------------------------------------------
    timming_end(">> pals_cpu_1pop_finalize", ts_finalize);
    // Timming -----------------------------------------------------
}


void rpals_cpu_init(struct params &input, struct etc_matrix *etc, struct energy_matrix *energy,
    int seed, struct rpals_cpu_instance &empty_instance)
{
    if (!OUTPUT_SOLUTION)
    {
        fprintf(stdout, "[INFO] == Input arguments =====================================\n");
        fprintf(stdout, "       Seed                                    : %d\n", seed);
        fprintf(stdout, "       Number of tasks                         : %d\n", etc->tasks_count);
        fprintf(stdout, "       Number of machines                      : %d\n", etc->machines_count);
        fprintf(stdout, "[INFO] == Configuration constants =============================\n");
        fprintf(stdout, "       RPALS_CPU_WORK__TIMEOUT                      : %d\n", input.max_time_secs);
        fprintf(stdout, "       RPALS_CPU_WORK__ITERATIONS                   : %d\n", input.max_iterations);
        fprintf(stdout, "[INFO] ========================================================\n");
    }

    // =========================================================================
    // Pido la memoria e inicializo la solucion de partida.

    empty_instance.max_time_secs = input.max_time_secs;
    empty_instance.max_iterations = input.max_iterations;

    empty_instance.etc = etc;
    empty_instance.energy = energy;

    empty_instance.best_solution = -1;
    empty_instance.total_iterations = 0;

    // Population.
    for (int i = 0; i < 2; i++)
    {
        empty_instance.population[i].status = SOLUTION__STATUS_EMPTY;
        empty_instance.population[i].initialized = 0;
    }
    
    // =========================================================================
    // Pedido de memoria para la generacion de numeros aleatorios.

    timespec ts_1;
    timming_start(ts_1);

    #ifdef CPU_MERSENNE_TWISTER
        cpu_mt_init(seed, empty_instance.random_state);
    #endif
    #ifdef CPU_RAND
        cpu_rand_init(seed, empty_instance.random_state);
    #endif
    #ifdef CPU_DRAND48
        cpu_drand48_init(seed, empty_instance.random_state);
    #endif

    timming_end(".. cpu_rand_buffers", ts_1);
}

void rpals_cpu_finalize(struct rpals_cpu_instance &instance)
{
    for (int i = 0; i < 2; i++)
    {
        if (instance.population[i].initialized == 1)
        {
            free_solution(&(instance.population[i]));
        }
    }
}
