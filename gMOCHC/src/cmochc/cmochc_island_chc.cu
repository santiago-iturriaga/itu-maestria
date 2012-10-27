#include <pthread.h>
#include <math.h>

#include "cmochc_island_pals.h"
#include "cmochc_island_chc.h"
#include "cmochc_island_utils.h"

void chc_population_init(int thread_id) {
    FLOAT random;

    for (int i = 0; i < MAX_POP_SOLS; i++) {
        EA_THREADS[thread_id].fitness_population[i] = NAN;

        if (i == 0) {
            COUNT_EVALUATIONS[thread_id]++;
            
            EA_THREADS[thread_id].makespan_zenith_value = EA_THREADS[thread_id].population[i].makespan;
            EA_THREADS[thread_id].makespan_nadir_value = EA_THREADS[thread_id].makespan_zenith_value;

            EA_THREADS[thread_id].energy_zenith_value = EA_THREADS[thread_id].population[i].energy_consumption;
            EA_THREADS[thread_id].energy_nadir_value = EA_THREADS[thread_id].energy_zenith_value;
        } else {
            // Random init.
            create_empty_solution(&(EA_THREADS[thread_id].population[i]));

            random = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]);

            int starting_pos;
            starting_pos = (int)(floor(INPUT.tasks_count * random));

            compute_mct_random(&(EA_THREADS[thread_id].population[i]), starting_pos, i & 0x1);
            //compute_minmin(&(EA_THREADS[thread_id].population[i]));

            COUNT_EVALUATIONS[thread_id]++;

            #ifdef CMOCHC_LOCAL__MUTATE_INITIAL_POP
                CHC__MUTATE(EA_INSTANCE.rand_state[thread_id], &EA_THREADS[thread_id].population[i], &EA_THREADS[thread_id].population[i])
                COUNT_EVALUATIONS[thread_id]++;
            #endif

            if (EA_THREADS[thread_id].population[i].makespan < EA_THREADS[thread_id].makespan_zenith_value) {
                EA_THREADS[thread_id].makespan_zenith_value = EA_THREADS[thread_id].population[i].makespan;
            }
            if (EA_THREADS[thread_id].population[i].makespan > EA_THREADS[thread_id].makespan_nadir_value) {
                EA_THREADS[thread_id].makespan_nadir_value = EA_THREADS[thread_id].population[i].makespan;
            }

            if (EA_THREADS[thread_id].population[i].energy_consumption < EA_THREADS[thread_id].energy_zenith_value) {
                EA_THREADS[thread_id].energy_zenith_value = EA_THREADS[thread_id].population[i].energy_consumption;
            }
            if (EA_THREADS[thread_id].population[i].energy_consumption > EA_THREADS[thread_id].energy_nadir_value) {
                EA_THREADS[thread_id].energy_nadir_value = EA_THREADS[thread_id].population[i].energy_consumption;
            }

            #ifdef DEBUG_3
                fprintf(stderr, "[DEBUG] Thread %d, solution=%d makespan=%.2f[z=%.2f|n=%.2f] energy=%.2f[z=%.2f|n=%.2f]\n",
                    thread_id, i,
                    EA_THREADS[thread_id].population[i].makespan,
                    EA_THREADS[thread_id].makespan_zenith_value,
                    EA_THREADS[thread_id].makespan_nadir_value,
                    EA_THREADS[thread_id].population[i].energy_consumption,
                    EA_THREADS[thread_id].energy_zenith_value,
                    EA_THREADS[thread_id].energy_nadir_value);
            #endif
        }
    }

    #ifdef DEBUG_3
        fprintf(stderr, "[DEBUG] Normalization references\n");
        fprintf(stderr, "> (makespan) zenith=%.2f, nadir=%.2f\n",
            EA_THREADS[thread_id].makespan_zenith_value, EA_THREADS[thread_id].makespan_nadir_value);
        fprintf(stderr, "> (energy) zenith=%.2f, nadir=%.2f\n",
            EA_THREADS[thread_id].energy_zenith_value, EA_THREADS[thread_id].energy_nadir_value);
    #endif

    for (int i = 0; i < MAX_POP_SOLS; i++) {
        EA_THREADS[thread_id].sorted_population[i] = i;
        EA_THREADS[thread_id].fitness_population[i] = NAN;
        fitness(thread_id, i);

        #ifdef DEBUG_3
            fprintf(stderr, "[DEBUG] Thread %d, solution=%d makespan=%.2f energy=%.2f fitness=%.2f\n",
                thread_id, i, EA_THREADS[thread_id].population[i].makespan,
                EA_THREADS[thread_id].population[i].energy_consumption,
                fitness(thread_id, i));
        #endif
    }
}

void chc_evolution(int thread_id) {
    FLOAT random;

    int ref_point_changed;
    int threshold;
    int next_avail_children;

    FLOAT d;
    int p1_idx, p2_idx;
    int p1_rand, p2_rand;
    int c1_idx, c2_idx;

    threshold = EA_THREADS[thread_id].threshold_max;
    ref_point_changed = 0;

    for (int iteracion = 0; iteracion < CMOCHC_LOCAL__ITERATION_COUNT; iteracion++) {
        #ifdef DEBUG_1
            COUNT_GENERATIONS[thread_id]++;
        #endif

        /* *********************************************************************************************
         * Mating
         * ********************************************************************************************* */
        next_avail_children = CMOCHC_LOCAL__POPULATION_SIZE;

        for (int child = 0; child < CMOCHC_LOCAL__POPULATION_SIZE / 2; child++) {
            if (next_avail_children + 1 < MAX_POP_SOLS) {
                if (RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]) < CHC__CROSS_PROB) {
                    // Padre aleatorio 1
                    random = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]);
                    p1_rand = (int)(floor(CMOCHC_LOCAL__POPULATION_SIZE * random));
                    p1_idx = EA_THREADS[thread_id].sorted_population[p1_rand];

                    // Padre aleatorio 2
                    random = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]);
                    p2_rand = (int)(floor((CMOCHC_LOCAL__POPULATION_SIZE - 1) * random));
                    if (p2_rand >= p1_rand) p2_rand++;
                    p2_idx = EA_THREADS[thread_id].sorted_population[p2_rand];

                    // Chequeo la distancia entre padres
                    d = CHC__DISTANCE(EA_INSTANCE.rand_state[thread_id],
                        &EA_THREADS[thread_id].population[p1_idx],
                        &EA_THREADS[thread_id].population[p2_idx]);
                    
                    /*d = distance(&EA_THREADS[thread_id].population[p1_idx],
                        &EA_THREADS[thread_id].population[p2_idx]);*/

                    if (d > threshold) {
                        // Aplico HUX y creo dos hijos
                        #if defined(DEBUG_1)
                            COUNT_CROSSOVER[thread_id]++;
                        #endif

                        c1_idx = EA_THREADS[thread_id].sorted_population[next_avail_children];
                        c2_idx = EA_THREADS[thread_id].sorted_population[next_avail_children+1];

                        CHC__CROSS(EA_INSTANCE.rand_state[thread_id],
                            &EA_THREADS[thread_id].population[p1_idx],&EA_THREADS[thread_id].population[p2_idx],
                            &EA_THREADS[thread_id].population[c1_idx],&EA_THREADS[thread_id].population[c2_idx])

                        COUNT_EVALUATIONS[thread_id]+=2;

                        EA_THREADS[thread_id].fitness_population[c1_idx] = NAN;
                        EA_THREADS[thread_id].fitness_population[c2_idx] = NAN;

                        // Evalúo el cambio en el pto. de referencia (Zenith/Nadir)
                        if (EA_THREADS[thread_id].population[p1_idx].makespan < EA_THREADS[thread_id].makespan_zenith_value) {
                            EA_THREADS[thread_id].makespan_zenith_value = EA_THREADS[thread_id].population[p1_idx].makespan;
                            ref_point_changed = 1;
                        }

                        if (EA_THREADS[thread_id].population[p1_idx].energy_consumption < EA_THREADS[thread_id].energy_zenith_value) {
                            EA_THREADS[thread_id].energy_zenith_value = EA_THREADS[thread_id].population[p1_idx].energy_consumption;
                            ref_point_changed = 1;
                        }

                        if (EA_THREADS[thread_id].population[p1_idx].makespan > EA_THREADS[thread_id].makespan_nadir_value) {
                            EA_THREADS[thread_id].makespan_nadir_value = EA_THREADS[thread_id].population[p1_idx].makespan;
                            ref_point_changed = 1;
                        }

                        if (EA_THREADS[thread_id].population[p1_idx].energy_consumption > EA_THREADS[thread_id].energy_nadir_value) {
                            EA_THREADS[thread_id].energy_nadir_value = EA_THREADS[thread_id].population[p1_idx].energy_consumption;
                            ref_point_changed = 1;
                        }

                        if (EA_THREADS[thread_id].population[p2_idx].makespan < EA_THREADS[thread_id].makespan_zenith_value) {
                            EA_THREADS[thread_id].makespan_zenith_value = EA_THREADS[thread_id].population[p2_idx].makespan;
                            ref_point_changed = 1;
                        }

                        if (EA_THREADS[thread_id].population[p2_idx].energy_consumption < EA_THREADS[thread_id].energy_zenith_value) {
                            EA_THREADS[thread_id].energy_zenith_value = EA_THREADS[thread_id].population[p2_idx].energy_consumption;
                            ref_point_changed = 1;
                        }

                        if (EA_THREADS[thread_id].population[p2_idx].makespan > EA_THREADS[thread_id].makespan_nadir_value) {
                            EA_THREADS[thread_id].makespan_nadir_value = EA_THREADS[thread_id].population[p2_idx].makespan;
                            ref_point_changed = 1;
                        }

                        if (EA_THREADS[thread_id].population[p2_idx].energy_consumption > EA_THREADS[thread_id].energy_nadir_value) {
                            EA_THREADS[thread_id].energy_nadir_value = EA_THREADS[thread_id].population[p2_idx].energy_consumption;
                            ref_point_changed = 1;
                        }

                        #ifdef DEBUG_1
                            // Si cambió el punto de referencia, actualizo todos los fitness
                            if (ref_point_changed == 0) {
                                fitness(thread_id, c1_idx);
                                fitness(thread_id, c2_idx);
                            } else {
                                ref_point_changed = 0;

                                fitness_reset(thread_id);
                                fitness_all(thread_id);
                            }

                            if ((EA_THREADS[thread_id].fitness_population[c1_idx] < EA_THREADS[thread_id].fitness_population[p1_idx])
                                ||(EA_THREADS[thread_id].fitness_population[c1_idx] < EA_THREADS[thread_id].fitness_population[p2_idx])
                                ||(EA_THREADS[thread_id].fitness_population[c2_idx] < EA_THREADS[thread_id].fitness_population[p1_idx])
                                ||(EA_THREADS[thread_id].fitness_population[c2_idx] < EA_THREADS[thread_id].fitness_population[p2_idx])) {

                                COUNT_IMPROVED_CROSSOVER[thread_id]++;
                            }
                        #endif
                    }

                    next_avail_children += 2;
                }
            }
        }

        if (next_avail_children != CMOCHC_LOCAL__POPULATION_SIZE) {
            // Si cambió el punto de referencia, actualizo todos los fitness
            if (ref_point_changed == 1) {
                ref_point_changed = 0;

                fitness_reset(thread_id);
                fitness_all(thread_id);
            } else {
                fitness_all(thread_id);
            }

            /* *********************************************************************************************
             * Sort parent+children population
             * ********************************************************************************************* */
            FLOAT best_parent;
            best_parent = fitness(thread_id, EA_THREADS[thread_id].sorted_population[0]);

            FLOAT worst_parent;
            worst_parent = fitness(thread_id, EA_THREADS[thread_id].sorted_population[CMOCHC_LOCAL__POPULATION_SIZE-1]);

            merge_sort(thread_id);

            if (worst_parent > fitness(thread_id, EA_THREADS[thread_id].sorted_population[CMOCHC_LOCAL__POPULATION_SIZE-1])) {
                #ifdef DEBUG_1
                    COUNT_AT_LEAST_ONE_CHILDREN_INSERTED[thread_id]++;
                #endif
            } else {
                threshold -= EA_THREADS[thread_id].threshold_step;
            }

            if (best_parent > fitness(thread_id, EA_THREADS[thread_id].sorted_population[0])) {
                #ifdef DEBUG_1
                    COUNT_IMPROVED_BEST_SOL[thread_id]++;
                #endif
            }
        } else {
            threshold -= EA_THREADS[thread_id].threshold_step;
        }

        /* Ejecuto la búsqueda local sobre una solución "elite" */
        int aux_index, pals_idx;

        //aux_index = 0;
        aux_index = (int)(RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]) * CMOCHC_LOCAL__BEST_SOLS_KEPT);
        //aux_index = (int)(RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]) * CMOCHC_LOCAL__POPULATION_SIZE);

        pals_idx = EA_THREADS[thread_id].sorted_population[aux_index];

        // Clono la mejor solución y descarto la peor solucion.
        clone_solution(&EA_THREADS[thread_id].population[EA_THREADS[thread_id].sorted_population[MAX_POP_SOLS-1]], &EA_THREADS[thread_id].population[pals_idx]);

        // Ejecuto el PALS sobre la copia de la mejor solucion.
        pals_search(thread_id, MAX_POP_SOLS-1);

        /* Re-sort de population */
        int current_pos = MAX_POP_SOLS-1;
        while ((current_pos > 0) &&
            (fitness(thread_id, EA_THREADS[thread_id].sorted_population[current_pos]) < fitness(thread_id, EA_THREADS[thread_id].sorted_population[current_pos-1]))) {

            aux_index = EA_THREADS[thread_id].sorted_population[current_pos];
            EA_THREADS[thread_id].sorted_population[current_pos] = EA_THREADS[thread_id].sorted_population[current_pos-1];
            EA_THREADS[thread_id].sorted_population[current_pos-1] = aux_index;
            current_pos--;
        }

        /*
        #ifdef DEBUG_3
            if (thread_id == 0) {
                fprintf(stderr, "[DEBUG] Current population:\n");
                for (int i = 0; i < CMOCHC_LOCAL__POPULATION_SIZE; i++) {
                    fprintf(stderr, "%.4f\n", fitness(thread_id, EA_THREADS[thread_id].sorted_population[i]));
                }
            }
        #endif
        * */

        if (threshold < 0) {
            threshold = EA_THREADS[thread_id].threshold_max;

            /* *********************************************************************************************
             * Cataclysm
             * ********************************************************************************************* */

            #ifdef DEBUG_1
                FLOAT pre_mut_fitness;
            #endif

            int current_index;
            ref_point_changed = 0;

            /* Muto el resto de las soluciones */
            for (int i = CMOCHC_LOCAL__BEST_SOLS_KEPT + 1; i < MAX_POP_SOLS; i++) { /* No muto las mejores soluciones */
                current_index = EA_THREADS[thread_id].sorted_population[i];

                assert(EA_THREADS[thread_id].population[current_index].initialized == SOLUTION__IN_USE);
                if (EA_THREADS[thread_id].population[current_index].initialized == SOLUTION__IN_USE) {
                    if (RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]) < CHC__MUTATION_PROB) {
                        #ifdef DEBUG_1
                            COUNT_CATACLYSM[thread_id]++;
                            pre_mut_fitness = fitness(thread_id, current_index);
                        #endif

                        aux_index = (int)(RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]) * CMOCHC_LOCAL__BEST_SOLS_KEPT);

                        /* Muto la solución */
                        CHC__MUTATE(EA_INSTANCE.rand_state[thread_id],
                            &EA_THREADS[thread_id].population[EA_THREADS[thread_id].sorted_population[aux_index]],
                            &EA_THREADS[thread_id].population[current_index])

                        COUNT_EVALUATIONS[thread_id]++;

                        EA_THREADS[thread_id].fitness_population[current_index] = NAN;

                        /* Chequeo si cambió el punto de referencia */
                        if (EA_THREADS[thread_id].population[current_index].makespan < EA_THREADS[thread_id].makespan_zenith_value) {
                            EA_THREADS[thread_id].makespan_zenith_value = EA_THREADS[thread_id].population[current_index].makespan;
                            ref_point_changed = 1;
                        }

                        if (EA_THREADS[thread_id].population[current_index].energy_consumption < EA_THREADS[thread_id].energy_zenith_value) {
                            EA_THREADS[thread_id].energy_zenith_value = EA_THREADS[thread_id].population[current_index].energy_consumption;
                            ref_point_changed = 1;
                        }

                        if (EA_THREADS[thread_id].population[current_index].makespan > EA_THREADS[thread_id].makespan_nadir_value) {
                            EA_THREADS[thread_id].makespan_nadir_value = EA_THREADS[thread_id].population[current_index].makespan;
                            ref_point_changed = 1;
                        }

                        if (EA_THREADS[thread_id].population[current_index].energy_consumption > EA_THREADS[thread_id].energy_nadir_value) {
                            EA_THREADS[thread_id].energy_nadir_value = EA_THREADS[thread_id].population[current_index].energy_consumption;
                            ref_point_changed = 1;
                        }

                        #ifdef DEBUG_1
                            if (ref_point_changed == 0) {
                                fitness(thread_id, current_index);
                            } else {
                                fitness_reset(thread_id);
                                fitness_all(thread_id);
                            }

                            if (EA_THREADS[thread_id].fitness_population[current_index] <= pre_mut_fitness) {
                                COUNT_IMPOVED_CATACLYSM[thread_id]++;
                            }
                        #endif
                    }
                }
            }

            // Si cambió el punto de referencia, actualizo todos los fitness
            if (ref_point_changed == 1) {
                ref_point_changed = 0;

                fitness_reset(thread_id);
                fitness_all(thread_id);
            } else {
                fitness_all(thread_id);
            }

            /* Re-sort de population */
            merge_sort(thread_id);
        }
    }
}
