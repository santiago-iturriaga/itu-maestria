#include <stdlib.h>
#include <stdio.h>

#include "../../random/cpu_rand.h"
#include "../../random/cpu_drand48.h"
#include "../../random/cpu_mt.h"

#include "adhoc.h"

int archivers_adhoc(struct pals_cpu_1pop_thread_arg *instance, int new_solution_pos)
{
    if (DEBUG_DEV) fprintf(stdout, "========================================================\n");
    double random = 0.0;

    float makespan_new, energy_new;
    makespan_new = get_makespan(&(instance->population[new_solution_pos]));
    energy_new = get_energy(&(instance->population[new_solution_pos]));

    if (*(instance->best_energy_solution) == -1) *(instance->best_energy_solution) = new_solution_pos;
    if (*(instance->best_makespan_solution) == -1) *(instance->best_makespan_solution) = new_solution_pos;

    float best_energy_value = get_energy(&(instance->population[*(instance->best_energy_solution)]));
    float best_makespan_value = get_makespan(&(instance->population[*(instance->best_makespan_solution)]));

    if (DEBUG_DEV)
    {
        fprintf(stdout, "[DEBUG] Population\n");
        fprintf(stdout, "        Population_count: %d\n", *(instance->population_count));
        fprintf(stdout, "        Solution to eval: %d\n", new_solution_pos);
        fprintf(stdout, "        Makespan        : %f\n", makespan_new);
        fprintf(stdout, "        Energy          : %f\n", energy_new);
        fprintf(stdout, "        Best makespan   : %f (%d)\n", best_makespan_value, *(instance->best_makespan_solution));
        fprintf(stdout, "        Best energy     : %f (%d)\n", best_energy_value, *(instance->best_energy_solution));

        for (int i = 0; i < instance->population_max_size; i++)
        {
            float makespan, energy;
            makespan = 0;
            energy = 0;
            
            if (instance->population[i].status == SOLUTION__STATUS_READY) {
                makespan = get_makespan(&(instance->population[i]));
                energy = get_energy(&(instance->population[i]));
            }

            fprintf(stdout, " >> sol.pos[%d] init=%d status=%d makespan=%f energy=%f\n", i,
                instance->population[i].initialized, instance->population[i].status,
                makespan, energy);
        }
    }

    int candidato_reemplazo = -1;
    int solutions_deleted = 0;
    int new_solution_is_dominated = 0;

    int s_idx = -1;
    for (int s_pos = 0; (s_pos < instance->population_max_size) && (new_solution_is_dominated == 0); s_pos++)
    {

        if ((instance->population[s_pos].status > SOLUTION__STATUS_EMPTY) &&
            (instance->population[s_pos].initialized == 1) &&
            (s_pos != new_solution_pos))
        {
            s_idx++;

            // Calculo no dominancia del elemento nuevo con el actual.
            float makespan, energy;
            makespan = get_makespan(&(instance->population[s_pos]));
            energy = get_energy(&(instance->population[s_pos]));

            if (DEBUG_DEV) fprintf(stdout, "[%d] Makespan: %f %f || Energy %f %f\n", s_pos, makespan, makespan_new, energy, energy_new);

            if ((makespan <= makespan_new) && (energy <= energy_new))
            {
                // La nueva solucion es dominada por una ya existente.
                new_solution_is_dominated = 1;

                if (DEBUG_DEV) fprintf(stdout, "[DEBUG] Individual %d is dominated by %d\n", new_solution_pos, s_pos);
            }
            else if ((makespan_new <= makespan) && (energy_new <= energy))
            {
                // La nueva solucin domina a una ya existente.
                solutions_deleted++;
                instance->population_count[0] = instance->population_count[0] - 1;
                instance->population[s_pos].status = SOLUTION__STATUS_EMPTY;

                if (DEBUG_DEV) fprintf(stdout, "[DEBUG] Removed individual %d because %d is better\n", s_pos, new_solution_pos);
            }
            else
            {
                if (DEBUG_DEV) fprintf(stdout, "[DEBUG] No definido\n");

                if ((instance->population_count[0] + instance->count_threads) >= instance->population_max_size) {
                    // Ninguna de las dos soluciones es dominada por la otra.

                    if ((*(instance->best_energy_solution) == s_pos) && (best_energy_value < energy_new)) {
                        // No lo puedo reemplazar porque es el mejor energy.

                    } else if ((*(instance->best_makespan_solution) == s_pos) && (best_makespan_value < makespan_new)) {
                        // No lo puedo reemplazar porque es el mejor makespan.

                    } else {
                        if (candidato_reemplazo == -1) {
                            candidato_reemplazo = s_pos;
                        } else {
                            float diff_makespan_candidato_actual;
                            float diff_energy_candidato_actual;
                            diff_makespan_candidato_actual = get_makespan(&(instance->population[candidato_reemplazo])) - makespan_new;
                            diff_energy_candidato_actual = get_energy(&(instance->population[candidato_reemplazo])) - energy_new;

                            float diff_makespan_individuo_actual;
                            float diff_energy_individuo_actual;
                            diff_makespan_individuo_actual = makespan - makespan_new;
                            diff_energy_individuo_actual = makespan - energy_new;

                            if (DEBUG_DEV) {
                                fprintf(stdout, "[ND] Evaluo candidato contra:\n");
                                fprintf(stdout, "[DEBUG] Makespan vs: %f vs %f (%f , %f)\n", get_makespan(&(instance->population[candidato_reemplazo])),
                                    get_makespan(&(instance->population[s_pos])),diff_makespan_candidato_actual, diff_makespan_individuo_actual);
                                fprintf(stdout, "[DEBUG] Energy vs: %f vs %f (%f , %f)\n", get_energy(&(instance->population[candidato_reemplazo])),
                                    get_energy(&(instance->population[s_pos])),diff_energy_candidato_actual, diff_energy_individuo_actual);
                            }

                            if (diff_makespan_individuo_actual > diff_makespan_candidato_actual) {
                                candidato_reemplazo = s_pos;

                            } else if ((diff_makespan_individuo_actual == diff_makespan_candidato_actual) &&
                                (diff_energy_individuo_actual > diff_energy_candidato_actual)) {

                                candidato_reemplazo = s_pos;

                            } else if ((diff_makespan_individuo_actual == diff_makespan_candidato_actual) &&
                                (diff_energy_individuo_actual == diff_energy_candidato_actual)) {

                                if (DEBUG_DEV) fprintf(stdout, "[ND] Sorteo un candidato.\n");

                                #ifdef CPU_MERSENNE_TWISTER
                                random = cpu_mt_generate(*(instance->thread_random_state));
                                #endif
                                #ifdef CPU_RAND
                                random = cpu_rand_generate(*(instance->thread_random_state));
                                #endif
                                #ifdef CPU_DRAND48
                                random = cpu_drand48_generate(*(instance->thread_random_state));
                                #endif

                                if (random > 0.5) {
                                    candidato_reemplazo = s_pos;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (new_solution_is_dominated == 0)
    {
        if ((instance->population_count[0] + instance->count_threads) < instance->population_max_size)
        {
            instance->population[new_solution_pos].status = SOLUTION__STATUS_READY;
            instance->population_count[0] = instance->population_count[0] + 1;

            if (energy_new < best_energy_value) {
                *(instance->best_energy_solution) = new_solution_pos;

                if (DEBUG_DEV) fprintf(stdout, "[DEBUG] New best energy solution %d\n", new_solution_pos);
            }
            if (makespan_new < best_makespan_value) {
                *(instance->best_makespan_solution) = new_solution_pos;

                if (DEBUG_DEV) fprintf(stdout, "[DEBUG] New best makespan solution %d\n", new_solution_pos);
            }

            if (DEBUG_DEV) fprintf(stdout, "[DEBUG] Added invidiual %d because is ND\n", new_solution_pos);
            return 1;
        }
        else
        {
            instance->total_population_full++;
            
            if (candidato_reemplazo != -1) {
                if (DEBUG_DEV) {
                    fprintf(stdout, "[DEBUG] Reemplazo por el individuo %d\n", candidato_reemplazo);
                    fprintf(stdout, "[DEBUG] Makespan vs: %f vs %f\n", get_makespan(&(instance->population[candidato_reemplazo])),
                        get_makespan(&(instance->population[new_solution_pos])));
                    fprintf(stdout, "[DEBUG] Energy vs: %f vs %f\n", get_energy(&(instance->population[candidato_reemplazo])),
                        get_energy(&(instance->population[new_solution_pos])));
                }

                instance->population[new_solution_pos].status = SOLUTION__STATUS_READY;
                instance->population[candidato_reemplazo].status = SOLUTION__STATUS_EMPTY;

                if (energy_new < best_energy_value) {
                    *(instance->best_energy_solution) = new_solution_pos;

                    if (DEBUG_DEV) fprintf(stdout, "[DEBUG] New best energy solution %d\n", new_solution_pos);
                } /*else {
                    assert(candidato_reemplazo != *(instance->best_energy_solution));
                }*/
                if (makespan_new < best_makespan_value) {
                    *(instance->best_makespan_solution) = new_solution_pos;

                    if (DEBUG_DEV) fprintf(stdout, "[DEBUG] New best makespan solution %d\n", new_solution_pos);
                } /*else {
                    assert(candidato_reemplazo != *(instance->best_makespan_solution));
                }*/

                return 1;
            } else {
                instance->population[new_solution_pos].status = SOLUTION__STATUS_EMPTY;

                if (DEBUG_DEV) fprintf(stdout, "[DEBUG] Discarded invidiual %d because there is no space left (threads=%d, count=%d, max=%d)\n",
                        new_solution_pos, instance->count_threads, instance->population_count[0], instance->population_max_size);
                return -1;
            }
        }
    }
    else
    {
        instance->population[new_solution_pos].status = SOLUTION__STATUS_EMPTY;

        if (DEBUG_DEV) fprintf(stdout, "[DEBUG] Discarded invidiual %d because is dominated\n", new_solution_pos);
        return 0;
    }
}
