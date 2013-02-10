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

#include "archivers/aga.h"
#include "me_mls_cpu.h"

void validate_thread_instance(struct pals_cpu_1pop_thread_arg *instance)
{
    pthread_mutex_lock(instance->population_mutex);
    int cantidad = 0;

    for (int i = 0; i < instance->population_max_size; i++)
    {
        if (instance->population[i].status > 0)
        {
            cantidad++;
        }
    }

    if (cantidad != *(instance->population_count))
    {
        fprintf(stdout, "[DEBUG] Population:\n");
        fprintf(stdout, "        Expected population count: %d\n", *(instance->population_count));
        fprintf(stdout, "        Real population count: %d\n", cantidad);
        for (int j = 0; j < instance->population_max_size; j++)
        {
            fprintf(stdout, "        [%d] status      %d\n", j, instance->population[j].status);
            fprintf(stdout, "        [%d] initialized %d\n", j, instance->population[j].initialized);
        }
    }

    pthread_mutex_unlock(instance->population_mutex);

    assert(cantidad == *(instance->population_count));
}

int pals_cpu_1pop_adhoc_arch(struct pals_cpu_1pop_thread_arg *instance, int new_solution_pos)
{
    #if defined(DEBUG_DEV)
    fprintf(stdout, "========================================================\n");
    #endif

    float makespan_new, energy_new;
    makespan_new = get_makespan(&(instance->population[new_solution_pos]));
    energy_new = get_energy(&(instance->population[new_solution_pos]));

    if (*(instance->best_energy_solution) == -1) *(instance->best_energy_solution) = new_solution_pos;
    if (*(instance->best_makespan_solution) == -1) *(instance->best_makespan_solution) = new_solution_pos;

    float best_energy_value = get_energy(&(instance->population[*(instance->best_energy_solution)]));
    float best_makespan_value = get_makespan(&(instance->population[*(instance->best_makespan_solution)]));

    #if defined(DEBUG_DEV)
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
        if (instance->population[i].status == 2) {
            makespan = get_makespan(&(instance->population[i]));
            energy = get_energy(&(instance->population[i]));
        }

        fprintf(stdout, " >> sol.pos[%d] init=%d status=%d makespan=%f energy=%f\n", i,
            instance->population[i].initialized, instance->population[i].status,
            makespan, energy);
    }
    #endif

    int candidato_reemplazo = -1;
    float candidato_reemplazo_improv = 0.0;
    
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

            #if defined(DEBUG_DEV)
            fprintf(stdout, "[%d] Makespan: %f %f || Energy %f %f\n", s_pos, makespan, makespan_new, energy, energy_new);
            #endif

            if ((makespan <= makespan_new) && (energy <= energy_new))
            {
                // La nueva solucion es dominada por una ya existente.
                new_solution_is_dominated = 1;

                #if defined(DEBUG_DEV)
                fprintf(stdout, "[DEBUG] Individual %d is dominated by %d\n", new_solution_pos, s_pos);
                #endif
            }
            else if ((makespan_new <= makespan) && (energy_new <= energy))
            {
                // La nueva solucin domina a una ya existente.
                solutions_deleted++;
                instance->population_count[0] = instance->population_count[0] - 1;
                instance->population[s_pos].status = SOLUTION__STATUS_EMPTY;

                #if defined(DEBUG_DEV)
                fprintf(stdout, "[DEBUG] Removed individual %d because %d is better\n", s_pos, new_solution_pos);
                #endif
            }
            else
            {
                #if defined(DEBUG_DEV)
                fprintf(stdout, "[DEBUG] No definido\n");
                #endif

                if ((instance->population_count[0] + instance->count_threads) >= instance->population_max_size) {
                    // Ninguna de las dos soluciones es dominada por la otra.

                    if ((*(instance->best_energy_solution) == s_pos) && (best_energy_value < energy_new)) {
                        // No lo puedo reemplazar porque es el mejor energy.

                    } else if ((*(instance->best_makespan_solution) == s_pos) && (best_makespan_value < makespan_new)) {
                        // No lo puedo reemplazar porque es el mejor makespan.

                    } else {
                        if (candidato_reemplazo == -1) {
                            candidato_reemplazo = s_pos;

                            float diff_makespan_candidato_actual;
                            float diff_energy_candidato_actual;
                            diff_makespan_candidato_actual = (get_makespan(&(instance->population[candidato_reemplazo])) - makespan_new) / makespan_new;
                            diff_energy_candidato_actual = (get_energy(&(instance->population[candidato_reemplazo])) - energy_new) / energy_new;
                            
                            candidato_reemplazo_improv = diff_makespan_candidato_actual + diff_energy_candidato_actual;
                        } else {
                            float diff_makespan_individuo_actual;
                            float diff_energy_individuo_actual;
                            diff_makespan_individuo_actual = (makespan - makespan_new) / makespan_new;
                            diff_energy_individuo_actual = (energy - energy_new) / energy_new;

                            #if defined(DEBUG_DEV)
                                fprintf(stdout, "[ND] Evaluo candidato contra:\n");
                                fprintf(stdout, "[DEBUG] Makespan vs: %f vs %f (%f , %f)\n", get_makespan(&(instance->population[candidato_reemplazo])),
                                    get_makespan(&(instance->population[s_pos])),diff_makespan_candidato_actual, diff_makespan_individuo_actual);
                                fprintf(stdout, "[DEBUG] Energy vs: %f vs %f (%f , %f)\n", get_energy(&(instance->population[candidato_reemplazo])),
                                    get_energy(&(instance->population[s_pos])),diff_energy_candidato_actual, diff_energy_individuo_actual);
                            #endif

                            if ((diff_makespan_individuo_actual + diff_energy_individuo_actual) > candidato_reemplazo_improv) {
                                candidato_reemplazo = s_pos;
                                candidato_reemplazo_improv = diff_makespan_individuo_actual + diff_energy_individuo_actual;
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

                #if defined(DEBUG_DEV)
                fprintf(stdout, "[DEBUG] New best energy solution %d\n", new_solution_pos);
                #endif
            }
            if (makespan_new < best_makespan_value) {
                *(instance->best_makespan_solution) = new_solution_pos;

                #if defined(DEBUG_DEV)
                fprintf(stdout, "[DEBUG] New best makespan solution %d\n", new_solution_pos);
                #endif
            }

            #if defined(DEBUG_DEV)
            fprintf(stdout, "[DEBUG] Added invidiual %d because is ND\n", new_solution_pos);
            #endif
            return 1;
        }
        else
        {
            if (candidato_reemplazo != -1) {
                #if defined(DEBUG_DEV)
                fprintf(stdout, "[DEBUG] Reemplazo por el individuo %d\n", candidato_reemplazo);
                fprintf(stdout, "[DEBUG] Makespan vs: %f vs %f\n", get_makespan(&(instance->population[candidato_reemplazo])),
                    get_makespan(&(instance->population[new_solution_pos])));
                fprintf(stdout, "[DEBUG] Energy vs: %f vs %f\n", get_energy(&(instance->population[candidato_reemplazo])),
                    get_energy(&(instance->population[new_solution_pos])));
                #endif

                instance->population[new_solution_pos].status = SOLUTION__STATUS_READY;
                instance->population[candidato_reemplazo].status = SOLUTION__STATUS_EMPTY;

                if (energy_new < best_energy_value) {
                    *(instance->best_energy_solution) = new_solution_pos;

                    #if defined(DEBUG_DEV)
                    fprintf(stdout, "[DEBUG] New best energy solution %d\n", new_solution_pos);
                    #endif
                }

                if (makespan_new < best_makespan_value) {
                    *(instance->best_makespan_solution) = new_solution_pos;

                    #if defined(DEBUG_DEV)
                    fprintf(stdout, "[DEBUG] New best makespan solution %d\n", new_solution_pos);
                    #endif
                }

                return 1;
            } else {
                instance->population[new_solution_pos].status = SOLUTION__STATUS_EMPTY;
                instance->total_population_full++;

                #if defined(DEBUG_DEV)
                fprintf(stdout, "[DEBUG] Discarded invidiual %d because there is no space left (threads=%d, count=%d, max=%d)\n",
                    new_solution_pos, instance->count_threads, instance->population_count[0], instance->population_max_size);
                #endif
                return -1;
            }
        }
    }
    else
    {
        instance->population[new_solution_pos].status = SOLUTION__STATUS_EMPTY;

        #if defined(DEBUG_DEV)
        fprintf(stdout, "[DEBUG] Discarded invidiual %d because is dominated\n", new_solution_pos);
        #endif
        return 0;
    }
}

void me_mls_cpu(struct params &input, struct etc_matrix *etc, struct energy_matrix *energy)
{
    // ==============================================================================
    // PALS aleatorio por tarea.
    // ==============================================================================

    // Timming -----------------------------------------------------
    timespec ts_init;
    timming_start(ts_init);
    // Timming -----------------------------------------------------

    timespec ts_total_time_start;
    clock_gettime(CLOCK_REALTIME, &ts_total_time_start);

    // Inicializo la memoria y los hilos de ejecucin.
    struct pals_cpu_1pop_instance instance;
    pals_cpu_1pop_init(input, etc, energy, input.seed, instance);

    // Timming -----------------------------------------------------
    timming_end(">> pals_cpu_1pop_init", ts_init);
    // Timming -----------------------------------------------------

    // Bloqueo la ejecucin hasta que terminen todos los hilos.
    for(int i = 0; i < instance.count_threads; i++)
    {
        if(pthread_join(instance.threads[i], NULL))
        {
            printf("Could not join thread %d\n", i);
            exit(EXIT_FAILURE);
        }
        else
        {
            #if defined(DEBUG)
            printf("[DEBUG] thread %d <OK>\n", i);
            #endif
        }
    }

    timespec ts_total_time_end;
    clock_gettime(CLOCK_REALTIME, &ts_total_time_end);

    // Timming -----------------------------------------------------
    timespec ts_finalize;
    timming_start(ts_finalize);
    // Timming -----------------------------------------------------

    // Todos los individuos que estaban para borrar al final no se borran.
    for (int i = 0; i < instance.population_max_size; i++)
    {
        if (instance.population[i].status == SOLUTION__STATUS_TO_DEL)
        {
            instance.population[i].status = SOLUTION__STATUS_READY;
        }
    }
    // ===========> DEBUG
    int total_iterations = 0;
    int total_makespan_greedy_searches = 0;
    int total_energy_greedy_searches = 0;
    int total_random_greedy_searches = 0;
    int total_success_makespan_greedy_searches = 0;
    int total_success_energy_greedy_searches = 0;
    int total_success_random_greedy_searches = 0;
    int total_swaps = 0;
    int total_moves = 0;
    int total_population_full = 0;
    int total_soluciones_no_evolucionadas = 0;
    int total_soluciones_evolucionadas_dominadas = 0;
    int total_re_iterations = 0;
    double elapsed_total_time = 0.0;
    double elapsed_last_found = 0.0;

    timespec ts_last_found = instance.threads_args[0].ts_last_found;

    for (int i = 0; i < instance.count_threads; i++)
    {
        total_iterations += instance.threads_args[i].total_iterations;
        total_makespan_greedy_searches += instance.threads_args[i].total_makespan_greedy_searches;
        total_energy_greedy_searches += instance.threads_args[i].total_energy_greedy_searches;
        total_random_greedy_searches += instance.threads_args[i].total_random_greedy_searches;
        total_success_makespan_greedy_searches += instance.threads_args[i].total_success_makespan_greedy_searches;
        total_success_energy_greedy_searches += instance.threads_args[i].total_success_energy_greedy_searches;
        total_success_random_greedy_searches += instance.threads_args[i].total_success_random_greedy_searches;
        total_swaps += instance.threads_args[i].total_swaps;
        total_moves += instance.threads_args[i].total_moves;
        total_population_full += instance.threads_args[i].total_population_full;
        total_soluciones_no_evolucionadas += instance.threads_args[i].total_soluciones_no_evolucionadas;
        total_soluciones_evolucionadas_dominadas += instance.threads_args[i].total_soluciones_evolucionadas_dominadas;
        total_re_iterations += instance.threads_args[i].total_re_iterations;

        if ((instance.threads_args[i].ts_last_found.tv_sec > ts_last_found.tv_sec) ||
            ((instance.threads_args[i].ts_last_found.tv_sec == ts_last_found.tv_sec) &&
            (instance.threads_args[i].ts_last_found.tv_nsec > ts_last_found.tv_nsec)))
        {

            ts_last_found = instance.threads_args[i].ts_last_found;
        }
    }

    elapsed_total_time = ((ts_total_time_end.tv_sec - ts_total_time_start.tv_sec) * 1000000.0) +
        ((ts_total_time_end.tv_nsec - ts_total_time_start.tv_nsec) / 1000.0);

    elapsed_last_found = ((ts_last_found.tv_sec - ts_total_time_start.tv_sec) * 1000000.0) +
        ((ts_last_found.tv_nsec - ts_total_time_start.tv_nsec) / 1000.0);

    if (!OUTPUT_SOLUTION)
    {
        fprintf(stdout, "[INFO] Cantidad de iteraciones             : %d\n", total_iterations);
        fprintf(stdout, "[INFO] Total de makespan searches          : %d (%d = %.1f)\n",
            total_makespan_greedy_searches, total_success_makespan_greedy_searches,
            (total_success_makespan_greedy_searches * 100.0 / total_makespan_greedy_searches));
        fprintf(stdout, "[INFO] Total de energy searches            : %d (%d = %.1f)\n",
            total_energy_greedy_searches, total_success_energy_greedy_searches,
            (total_success_energy_greedy_searches * 100.0 / total_energy_greedy_searches));
        fprintf(stdout, "[INFO] Total de random searches            : %d (%d = %.1f)\n",
            total_random_greedy_searches, total_success_random_greedy_searches,
            (total_success_random_greedy_searches * 100.0 / total_random_greedy_searches));
        fprintf(stdout, "[INFO] Total de swaps                      : %d\n", total_swaps);
        fprintf(stdout, "[INFO] Total de moves                      : %d\n", total_moves);
        fprintf(stdout, "[INFO] Total poblacion llena               : %d\n", total_population_full);
        fprintf(stdout, "[INFO] Cantidad de soluciones              : %d\n", instance.population_count);
        fprintf(stdout, "[INFO] Cantidad de soluciones no mejoradas : %d\n", total_soluciones_no_evolucionadas);
        fprintf(stdout, "[INFO] Cantidad de soluciones dominadas    : %d\n", total_soluciones_evolucionadas_dominadas);
        fprintf(stdout, "[INFO] Cantidad de re-trabajos             : %d\n", total_re_iterations);
        fprintf(stdout, "[INFO] Total execution time                : %.0f\n", elapsed_total_time);
        fprintf(stdout, "[INFO] Last solution found                 : %.0f\n", elapsed_last_found);

        fprintf(stdout, "== Threads ====================================================\n");
        for (int i = 0; i < instance.count_threads; i++)
        {
            fprintf(stdout, "Thread[%d] >> Iterations = %d >> Last found %d\n", i, instance.threads_args[i].total_iterations,
                instance.threads_args[i].iter_last_found);
        }
        #if defined(DEBUG_DEV)
            for (int i = 0; i < instance.population_max_size; i++)
            {
                if (instance.population[i].status == SOLUTION__STATUS_READY)
                {
                    validate_solution(&(instance.population[i]));
                }
            }
        #endif
    }
    // <=========== DEBUG

    #if defined(DEBUG)
    fprintf(stdout, "== Population =================================================\n");
    for (int i = 0; i < instance.population_max_size; i++)
    {
        fprintf(stdout, "%f %f (%d)\n", get_makespan(&(instance.population[i])), get_energy(&(instance.population[i])),
            instance.population[i].status);
    }
    #else
    if (!OUTPUT_SOLUTION)
    {
        fprintf(stdout, "== Population =================================================\n");
        for (int i = 0; i < instance.population_max_size; i++)
        {
                fprintf(stdout, "%f %f (%d)\n", get_makespan(&(instance.population[i])), get_energy(&(instance.population[i])),
                    instance.population[i].status);
        }
    }
    else
    {
        fprintf(stdout, "%d\n", instance.population_count);
        for (int i = 0; i < instance.population_max_size; i++)
        {
            if (instance.population[i].status > SOLUTION__STATUS_EMPTY)
            {
                for (int task = 0; task < etc->tasks_count; task++)
                {
                    fprintf(stdout, "%d\n", get_task_assigned_machine_id(&(instance.population[i]), task));
                }
            }
        }

        fprintf(stderr, "CANT_ITERACIONES|%d\n", total_iterations);
        fprintf(stderr, "TOTAL_TIME|%.0f\n", elapsed_total_time);
        fprintf(stderr, "LAST_FOUND_TIME|%.0f\n", elapsed_last_found);
        fprintf(stderr, "TOTAL_SWAPS|%d\n", total_swaps);
        fprintf(stderr, "TOTAL_MOVES|%d\n", total_moves);
        fprintf(stderr, "TOTAL_RANDOM_SEARCHES|%d\n", total_random_greedy_searches);
        fprintf(stderr, "TOTAL_ENERGY_SEARCHES|%d\n", total_energy_greedy_searches);
        fprintf(stderr, "TOTAL_MAKESPAN_SEARCHES|%d\n", total_makespan_greedy_searches);
        fprintf(stderr, "TOTAL_SUCCESS_RANDOM_SEARCHES|%d\n", total_success_random_greedy_searches);
        fprintf(stderr, "TOTAL_SUCCESS_ENERGY_SEARCHES|%d\n", total_success_energy_greedy_searches);
        fprintf(stderr, "TOTAL_SUCCESS_MAKESPAN_SEARCHES|%d\n", total_success_makespan_greedy_searches);
        fprintf(stderr, "TOTAL_POPULATION_FULL|%d\n", total_population_full);
        fprintf(stderr, "TOTAL_SOLS_NO_EVOLUCIONADAS|%d\n", total_soluciones_no_evolucionadas);
        fprintf(stderr, "TOTAL_SOLS_DOMINADAS|%d\n", total_soluciones_evolucionadas_dominadas);
        fprintf(stderr, "TOTAL_RE_TRABAJO|%d\n", total_re_iterations);
    }
    #endif

    // Libero la memoria del dispositivo.
    pals_cpu_1pop_finalize(instance);

    // Timming -----------------------------------------------------
    timming_end(">> pals_cpu_1pop_finalize", ts_finalize);
    // Timming -----------------------------------------------------
}


void pals_cpu_1pop_init(struct params &input, struct etc_matrix *etc, struct energy_matrix *energy,
int seed, struct pals_cpu_1pop_instance &empty_instance)
{

    // Asignacin del paralelismo del algoritmo.
    empty_instance.count_threads = input.thread_count;

    if (!OUTPUT_SOLUTION)
    {
        fprintf(stdout, "[INFO] == Input arguments =====================================\n");
        fprintf(stdout, "       Seed                                    : %d\n", seed);
        fprintf(stdout, "       Number of tasks                         : %d\n", etc->tasks_count);
        fprintf(stdout, "       Number of machines                      : %d\n", etc->machines_count);
        fprintf(stdout, "       Number of threads                       : %d\n", empty_instance.count_threads);
        fprintf(stdout, "       Population size                         : %d\n", input.population_size);
        fprintf(stdout, "[INFO] == Configuration constants =============================\n");
        fprintf(stdout, "       PALS_CPU_1POP_WORK__TIMEOUT                      : %d\n", input.max_time_secs);
        fprintf(stdout, "       PALS_CPU_1POP_WORK__ITERATIONS                   : %d\n", input.max_iterations);
        fprintf(stdout, "       PALS_CPU_1POP_SEARCH_OP_BALANCE__SWAP            : %f\n", PALS_CPU_1POP_SEARCH_OP_BALANCE__SWAP);
        fprintf(stdout, "       PALS_CPU_1POP_SEARCH_OP_BALANCE__MOVE            : %f\n", PALS_CPU_1POP_SEARCH_OP_BALANCE__MOVE);
        fprintf(stdout, "       PALS_CPU_1POP_SEARCH_BALANCE__MAKESPAN           : %f\n", PALS_CPU_1POP_SEARCH_BALANCE__MAKESPAN);
        fprintf(stdout, "       PALS_CPU_1POP_SEARCH_BALANCE__ENERGY             : %f\n", PALS_CPU_1POP_SEARCH_BALANCE__ENERGY);
        fprintf(stdout, "       PALS_CPU_1POP_SEARCH__MAKESPAN_GREEDY_PSEL_BEST  : %f\n", PALS_CPU_1POP_SEARCH__MAKESPAN_GREEDY_PSEL_BEST);
        fprintf(stdout, "       PALS_CPU_1POP_SEARCH__MAKESPAN_GREEDY_PSEL_WORST : %f\n", PALS_CPU_1POP_SEARCH__MAKESPAN_GREEDY_PSEL_WORST);
        fprintf(stdout, "       PALS_CPU_1POP_SEARCH__ENERGY_GREEDY_PSEL_BEST    : %f\n", PALS_CPU_1POP_SEARCH__ENERGY_GREEDY_PSEL_BEST);
        fprintf(stdout, "       PALS_CPU_1POP_SEARCH__ENERGY_GREEDY_PSEL_WORST   : %f\n", PALS_CPU_1POP_SEARCH__ENERGY_GREEDY_PSEL_WORST);
        fprintf(stdout, "       PALS_CPU_1POP_WORK__THREAD_ITERATIONS            : %d\n", PALS_CPU_1POP_WORK__THREAD_ITERATIONS);
        fprintf(stdout, "       PALS_CPU_1POP_WORK__THREAD_RE_WORK_FACTOR        : %d\n", PALS_CPU_1POP_WORK__THREAD_RE_WORK_FACTOR);
        fprintf(stdout, "       PALS_CPU_1POP_WORK__SRC_TASK_NHOOD               : %d\n", PALS_CPU_1POP_WORK__SRC_TASK_NHOOD);
        fprintf(stdout, "       PALS_CPU_1POP_WORK__DST_TASK_NHOOD               : %d\n", PALS_CPU_1POP_WORK__DST_TASK_NHOOD);
        fprintf(stdout, "       PALS_CPU_1POP_WORK__DST_MACH_NHOOD               : %d\n", PALS_CPU_1POP_WORK__DST_MACH_NHOOD);
        fprintf(stdout, "[INFO] ========================================================\n");
    }

    assert(input.population_size > empty_instance.count_threads);

    // =========================================================================
    // Pido la memoria e inicializo la solucin de partida.

    empty_instance.etc = etc;
    empty_instance.energy = energy;

    empty_instance.population_max_size = input.population_size;
    empty_instance.population_count = 0;
    empty_instance.best_makespan_solution = -1;
    empty_instance.best_energy_solution = -1;
    empty_instance.global_total_iterations = 0;

    // Population.
    empty_instance.population = (struct solution*)malloc(sizeof(struct solution) * empty_instance.population_max_size);
    if (empty_instance.population == NULL)
    {
        fprintf(stderr, "[ERROR] Solicitando memoria para population.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < empty_instance.population_max_size; i++)
    {
        empty_instance.population[i].status = SOLUTION__STATUS_EMPTY;
        empty_instance.population[i].initialized = 0;
    }
    
    #if defined(ARCHIVER_AGA)
        if (!OUTPUT_SOLUTION) fprintf(stdout, "[INFO] Using AGA archiver\n");
        empty_instance.archiver_state = (struct aga_state*)malloc(sizeof(struct aga_state));

        archivers_aga_init(&empty_instance);
    #endif

    // =========================================================================
    // Pedido de memoria para la generacin de numeros aleatorios.

    timespec ts_1;
    timming_start(ts_1);

    srand(seed);
    long int random_seed;

    #ifdef CPU_MERSENNE_TWISTER
    empty_instance.random_states = (struct cpu_mt_state*)malloc(sizeof(struct cpu_mt_state) * empty_instance.count_threads);
    #endif
    #ifdef CPU_RAND
    empty_instance.random_states = (struct cpu_rand_state*)malloc(sizeof(struct cpu_rand_state) * empty_instance.count_threads);
    #endif
    #ifdef CPU_DRAND48
    empty_instance.random_states = (struct cpu_drand48_state*)malloc(sizeof(struct cpu_drand48_state) * empty_instance.count_threads);
    #endif

    for (int i = 0; i < empty_instance.count_threads; i++)
    {
        random_seed = rand();

        #ifdef CPU_MERSENNE_TWISTER
        cpu_mt_init(random_seed, empty_instance.random_states[i]);
        #endif
        #ifdef CPU_RAND
        cpu_rand_init(random_seed, empty_instance.random_states[i]);
        #endif
        #ifdef CPU_DRAND48
        cpu_drand48_init(random_seed, empty_instance.random_states[i]);
        #endif
    }

    timming_end(".. cpu_rand_buffers", ts_1);

    #if defined(INIT_PMINMIN)
        fprintf(stderr, "INIT|pMin-Min\n");
        init_empty_solution(etc, energy, &(empty_instance.population[0]));
        compute_pminmin(etc, &(empty_instance.population[0]), empty_instance.count_threads);
    #endif
                    
    // =========================================================================
    // Creo e inicializo los threads y los mecanismos de sincronizacin del sistema.

    timespec ts_threads;
    timming_start(ts_threads);

    if (pthread_mutex_init(&(empty_instance.population_mutex), NULL))
    {
        printf("Could not create a population mutex\n");
        exit(EXIT_FAILURE);
    }

    if (pthread_barrier_init(&(empty_instance.sync_barrier), NULL, empty_instance.count_threads))
    {
        printf("Could not create a sync barrier\n");
        exit(EXIT_FAILURE);
    }

    // Creo los hilos.
    empty_instance.threads = (pthread_t*)
        malloc(sizeof(pthread_t) * empty_instance.count_threads);

    empty_instance.threads_args = (struct pals_cpu_1pop_thread_arg*)
        malloc(sizeof(struct pals_cpu_1pop_thread_arg) * empty_instance.count_threads);

    empty_instance.work_type = PALS_CPU_1POP_WORK__INIT;

    timespec ts_start;
    clock_gettime(CLOCK_REALTIME, &ts_start);

    for (int i = 0; i < empty_instance.count_threads; i++)
    {
        empty_instance.threads_args[i].thread_idx = i;
        empty_instance.threads_args[i].count_threads = empty_instance.count_threads;

        empty_instance.threads_args[i].etc = empty_instance.etc;
        empty_instance.threads_args[i].energy = empty_instance.energy;

        empty_instance.threads_args[i].max_iterations = input.max_iterations;
        empty_instance.threads_args[i].max_time_secs = input.max_time_secs;

        empty_instance.threads_args[i].population = empty_instance.population;
        empty_instance.threads_args[i].population_count = &(empty_instance.population_count);
        empty_instance.threads_args[i].population_max_size = empty_instance.population_max_size;
        empty_instance.threads_args[i].best_makespan_solution = &(empty_instance.best_makespan_solution);
        empty_instance.threads_args[i].best_energy_solution = &(empty_instance.best_energy_solution);

        empty_instance.threads_args[i].archiver_state = empty_instance.archiver_state;
        empty_instance.threads_args[i].work_type = &(empty_instance.work_type);
        empty_instance.threads_args[i].global_total_iterations = &(empty_instance.global_total_iterations);

        empty_instance.threads_args[i].population_mutex = &(empty_instance.population_mutex);
        empty_instance.threads_args[i].sync_barrier = &(empty_instance.sync_barrier);

        empty_instance.threads_args[i].thread_random_state = &(empty_instance.random_states[i]);
        empty_instance.threads_args[i].ts_start = ts_start;

        if (pthread_create(&(empty_instance.threads[i]), NULL, pals_cpu_1pop_thread,  (void*) &(empty_instance.threads_args[i])))
        {
            printf("Could not create slave thread %d\n", i);
            exit(EXIT_FAILURE);
        }
    }

    timming_end(".. thread creation", ts_threads);
}


void pals_cpu_1pop_finalize(struct pals_cpu_1pop_instance &instance)
{
    for (int i = 0; i < instance.population_max_size; i++)
    {
        if (instance.population[i].initialized == 1)
        {
            free_solution(&(instance.population[i]));
        }
    }

    free(instance.population);
    free(instance.random_states);
    free(instance.threads);
    free(instance.threads_args);

    #if defined(ARCHIVER_AGA)
        archivers_aga_free(&instance);
    #endif
    
    pthread_mutex_destroy(&(instance.population_mutex));
    pthread_barrier_destroy(&(instance.sync_barrier));
}


void* pals_cpu_1pop_thread(void *thread_arg)
{
    int rc;

    struct pals_cpu_1pop_thread_arg *thread_instance;
    thread_instance = (pals_cpu_1pop_thread_arg*)thread_arg;

    thread_instance->total_iterations = 0;
    thread_instance->total_makespan_greedy_searches = 0;
    thread_instance->total_energy_greedy_searches = 0;
    thread_instance->total_random_greedy_searches = 0;
    thread_instance->total_swaps = 0;
    thread_instance->total_moves = 0;
    thread_instance->total_population_full = 0;
    thread_instance->total_success_makespan_greedy_searches = 0;
    thread_instance->total_success_energy_greedy_searches = 0;
    thread_instance->total_success_random_greedy_searches = 0;
    thread_instance->total_soluciones_no_evolucionadas = 0;
    thread_instance->total_soluciones_evolucionadas_dominadas = 0;
    thread_instance->total_re_iterations = 0;
    thread_instance->iter_last_found = 0;

    int terminate = 0;
    int work_type = -1;

    timespec ts_current;
    clock_gettime(CLOCK_REALTIME, &ts_current);

    thread_instance->ts_last_found = ts_current;

    int selected_solution_pos = -1;
    int local_iteration_count = 0;

    int candidate_to_del_pos;
    int i;
    int random_sol_index;
    int current_sol_pos;
    int current_sol_index;
    float original_makespan;
    float original_energy;
    int search_type;
    double search_type_random;
    int work_do_iteration;
    int work_iteration_size;
    int search_iteration;
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
    int mutex_locked;
    int new_solution_eval;

    double random = 0.0; // Variable random multi-proposito :)

    while ((terminate == 0) &&
        (ts_current.tv_sec - thread_instance->ts_start.tv_sec < thread_instance->max_time_secs) &&
        (thread_instance->total_iterations < thread_instance->max_iterations) &&
        (thread_instance->global_total_iterations[0] < thread_instance->max_iterations))
    {
        work_type = *(thread_instance->work_type);

        #if defined(DEBUG_DEV)
        printf("[DEBUG] [THREAD=%d] Work type = %d\n", thread_instance->thread_idx, work_type);
        #endif

        if (work_type == PALS_CPU_1POP_WORK__EXIT)
        {
            // PALS_CPU_1POP_WORK__EXIT =======================================================================
            // Finalizo la ejecucin del algoritmo!
            terminate = 1;

        }
        else if (work_type == PALS_CPU_1POP_WORK__INIT)
        {
            // PALS_CPU_1POP_WORK__INIT_POP ===================================================================

            // Timming -----------------------------------------------------
            timespec ts_mct;
            timming_start(ts_mct);
            // Timming -----------------------------------------------------

            if (thread_instance->thread_idx < (thread_instance->population_max_size - thread_instance->count_threads))
            {
                /*pthread_mutex_lock(thread_instance->population_mutex);
                    thread_instance->population[thread_instance->thread_idx].status = SOLUTION__STATUS_NOT_READY;
                pthread_mutex_unlock(thread_instance->population_mutex);*/

                if (thread_instance->thread_idx == 0)
                {
                    #if defined(INIT_PMINMIN)

                    #endif
                    #if defined(INIT_MINMIN)
                        // Inicializo el individuo que me toca.
                        init_empty_solution(thread_instance->etc, thread_instance->energy, &(thread_instance->population[thread_instance->thread_idx]));

                        fprintf(stderr, "INIT|MinMin\n");

                        compute_minmin(&(thread_instance->population[thread_instance->thread_idx]));
                    #endif
                    #if defined(INIT_MCT)
                        // Inicializo el individuo que me toca.
                        init_empty_solution(thread_instance->etc, thread_instance->energy, &(thread_instance->population[thread_instance->thread_idx]));

                        fprintf(stderr, "INIT|MCT\n");

                        compute_mct(&(thread_instance->population[thread_instance->thread_idx]));
                    #endif
                } else {
                    // Inicializo el individuo que me toca.
                    init_empty_solution(thread_instance->etc, thread_instance->energy, &(thread_instance->population[thread_instance->thread_idx]));

                    #ifdef CPU_MERSENNE_TWISTER
                        random = cpu_mt_generate(*(thread_instance->thread_random_state));
                    #endif
                    #ifdef CPU_RAND
                        random = cpu_rand_generate(*(thread_instance->thread_random_state));
                    #endif
                    #ifdef CPU_DRAND48
                        random = cpu_drand48_generate(*(thread_instance->thread_random_state));
                    #endif

                    int random_task = (int)floor(random * thread_instance->etc->tasks_count);
                    compute_custom_mct(&(thread_instance->population[thread_instance->thread_idx]), random_task);
                }
                
                pthread_mutex_lock(thread_instance->population_mutex);

                #if defined(ARCHIVER_AGA)
                    archivers_aga(thread_instance, thread_instance->thread_idx);
                #else
                    pals_cpu_1pop_adhoc_arch(thread_instance, thread_instance->thread_idx);
                #endif

                #if defined(DEBUG_DEV)
                fprintf(stdout, "[DEBUG] Population\n");
                fprintf(stdout, "        Population_count: %d\n", *(thread_instance->population_count));

                for (int i = 0; i < thread_instance->population_max_size; i++)
                {
                    fprintf(stdout, " >> sol.pos[%d] init=%d status=%d\n", i,
                        thread_instance->population[i].initialized,
                        thread_instance->population[i].status);
                }
                #endif

                pthread_mutex_unlock(thread_instance->population_mutex);

                #if defined(DEBUG)
                fprintf(stdout, "[DEBUG] Initializing individual %d (%f %f)\n",
                    thread_instance->thread_idx, get_makespan(&(thread_instance->population[thread_instance->thread_idx])),
                    get_energy(&(thread_instance->population[thread_instance->thread_idx])));
                #endif

                // Timming -----------------------------------------------------
                timming_end(">> Random MCT Time", ts_mct);
                // Timming -----------------------------------------------------

                #if defined(DEBUG_DEV)
                validate_solution(&(thread_instance->population[thread_instance->thread_idx]));
                #endif
            }

            // Espero a que los demas hilos terminen.
            rc = pthread_barrier_wait(thread_instance->sync_barrier);
            if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
            {
                printf("Could not wait on barrier\n");
                exit(EXIT_FAILURE);
            }

            // Comienza la bsqueda.
            pthread_mutex_lock(thread_instance->population_mutex);
                thread_instance->work_type[0] = PALS_CPU_1POP_WORK__SEARCH;
            pthread_mutex_unlock(thread_instance->population_mutex);

        }
        else if (work_type == PALS_CPU_1POP_WORK__SEARCH)
        {
            // PALS_CPU_1POP_WORK__SEARCH ====================================================================

            // Busco un lugar libre en la poblacin para clonar un individuo y evolucionarlo ==================
            candidate_to_del_pos = -1;
            if (selected_solution_pos == -1) {
                pthread_mutex_lock(thread_instance->population_mutex);

                    for (i = 0; (i < thread_instance->population_max_size) && (selected_solution_pos == -1); i++)
                    {
                        if (thread_instance->population[i].status == SOLUTION__STATUS_EMPTY)
                        {
                            thread_instance->population[i].status = SOLUTION__STATUS_NOT_READY;
                            selected_solution_pos = i;

                            #if defined(DEBUG_DEV) 
                                fprintf(stdout, "[DEBUG] Found individual %d free\n", selected_solution_pos);
                            #endif
                        } else if (thread_instance->population[i].status == SOLUTION__STATUS_TO_DEL) {
                            candidate_to_del_pos = i;
                        }
                    }

                    if ((selected_solution_pos == -1)&&(candidate_to_del_pos != -1)) {
                        selected_solution_pos = candidate_to_del_pos;
                        thread_instance->population[selected_solution_pos].status = SOLUTION__STATUS_NOT_READY;
                    }
                pthread_mutex_unlock(thread_instance->population_mutex);
            }

            // Si no encuentro un lugar libre? duermo un rato y vuelvo a probar?
            if (selected_solution_pos == -1)
            {
                // No se que hacer... panico! termino!
                fprintf(stdout, "[ERROR] Hilo finalizado.");

                terminate = 1;
                thread_instance->total_population_full++;

            }
            else
            {
                struct solution *selected_solution;
                selected_solution = &(thread_instance->population[selected_solution_pos]);

                // Si es necesario inicializo el individuo.
                if (selected_solution->initialized == 0)
                {
                    #if defined(DEBUG_DEV)
                        fprintf(stdout, "[DEBUG] Initializing individual %d\n", selected_solution_pos);
                    #endif
                    init_empty_solution(thread_instance->etc, thread_instance->energy, selected_solution);
                }

                // Sorteo la solucion con la que me toca trabajar  =====================================================
                #ifdef CPU_MERSENNE_TWISTER
                    random = cpu_mt_generate(*(thread_instance->thread_random_state));
                #endif
                #ifdef CPU_RAND
                    random = cpu_rand_generate(*(thread_instance->thread_random_state));
                #endif
                #ifdef CPU_DRAND48
                    random = cpu_drand48_generate(*(thread_instance->thread_random_state));
                #endif

                pthread_mutex_lock(thread_instance->population_mutex);
                    thread_instance->global_total_iterations[0] += local_iteration_count;
                    local_iteration_count = 0;

                    random_sol_index = (int)floor(random * (*(thread_instance->population_count)));

                    #if defined(DEBUG_DEV)
                    fprintf(stdout, "[DEBUG] Random selection\n");
                    fprintf(stdout, "        Population_count: %d\n", *(thread_instance->population_count));
                    fprintf(stdout, "        Random          : %f\n", random);
                    fprintf(stdout, "        Random_sol_index: %d\n", random_sol_index);

                    for (i = 0; i < thread_instance->population_max_size; i++)
                    {
                        fprintf(stdout, " >> sol.pos[%d] init=%d status=%d\n", i,
                            thread_instance->population[i].initialized,
                            thread_instance->population[i].status);
                    }
                    #endif

                    current_sol_pos = -1;
                    current_sol_index = -1;

                    for (i = 0; (i < thread_instance->population_max_size) && (current_sol_pos == -1); i++)
                    {
                        if (thread_instance->population[i].status > SOLUTION__STATUS_EMPTY)
                        {
                            current_sol_index++;

                            if (current_sol_index == random_sol_index)
                            {
                                current_sol_pos = i;
                            }
                        }
                    }

                    // Clono la solucion elegida =====================================================
                    #if defined(DEBUG_DEV)
                        fprintf(stdout, "[DEBUG] Cloning individual %d to %d\n", current_sol_pos, selected_solution_pos);
                    #endif
                    clone_solution(selected_solution, &(thread_instance->population[current_sol_pos]), 0);

                pthread_mutex_unlock(thread_instance->population_mutex);

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

                #ifdef CPU_MERSENNE_TWISTER
                    random = cpu_mt_generate(*(thread_instance->thread_random_state));
                #endif
                #ifdef CPU_RAND
                    random = cpu_rand_generate(*(thread_instance->thread_random_state));
                #endif
                #ifdef CPU_DRAND48
                    random = cpu_drand48_generate(*(thread_instance->thread_random_state));
                #endif

                search_type_random = 0.0;

                #ifdef CPU_MERSENNE_TWISTER
                    search_type_random = cpu_mt_generate(*(thread_instance->thread_random_state));
                #endif
                #ifdef CPU_RAND
                    search_type_random = cpu_rand_generate(*(thread_instance->thread_random_state));
                #endif
                #ifdef CPU_DRAND48
                    search_type_random = cpu_drand48_generate(*(thread_instance->thread_random_state));
                #endif

                if (search_type_random < PALS_CPU_1POP_SEARCH_BALANCE__MAKESPAN)
                {
                    search_type = PALS_CPU_1POP_SEARCH__MAKESPAN_GREEDY;
                    thread_instance->total_makespan_greedy_searches++;
                }
                else if (search_type_random < PALS_CPU_1POP_SEARCH_BALANCE__MAKESPAN + PALS_CPU_1POP_SEARCH_BALANCE__ENERGY)
                {
                    search_type = PALS_CPU_1POP_SEARCH__ENERGY_GREEDY;
                    thread_instance->total_energy_greedy_searches++;
                }
                else
                {
                    search_type = PALS_CPU_1POP_SEARCH__RANDOM_GREEDY;
                    thread_instance->total_random_greedy_searches++;
                }

                work_do_iteration = 1;
                work_iteration_size = (int)floor((PALS_CPU_1POP_WORK__THREAD_ITERATIONS / PALS_CPU_1POP_WORK__THREAD_RE_WORK_FACTOR) +
                    (random * (PALS_CPU_1POP_WORK__THREAD_ITERATIONS - (PALS_CPU_1POP_WORK__THREAD_ITERATIONS / PALS_CPU_1POP_WORK__THREAD_RE_WORK_FACTOR))));

                while (work_do_iteration == 1) {
                    work_do_iteration = 0;

                    for (search_iteration = 0; search_iteration < work_iteration_size; search_iteration++)
                    {
                        thread_instance->total_iterations++;
                        local_iteration_count++;

                        // Determino las maquinas de inicio para la busqueda.
                        if (search_type == PALS_CPU_1POP_SEARCH__MAKESPAN_GREEDY)
                        {
                            //fprintf(stdout, "[DEBUG] Makespan greedy\n");

                            #ifdef CPU_MERSENNE_TWISTER
                                random = cpu_mt_generate(*(thread_instance->thread_random_state));
                            #endif
                            #ifdef CPU_RAND
                                random = cpu_rand_generate(*(thread_instance->thread_random_state));
                            #endif
                            #ifdef CPU_DRAND48
                                random = cpu_drand48_generate(*(thread_instance->thread_random_state));
                            #endif

                            if (random > PALS_CPU_1POP_SEARCH__MAKESPAN_GREEDY_PSEL_WORST) {
                                #ifdef CPU_MERSENNE_TWISTER
                                    random = cpu_mt_generate(*(thread_instance->thread_random_state));
                                #endif
                                #ifdef CPU_RAND
                                    random = cpu_rand_generate(*(thread_instance->thread_random_state));
                                #endif
                                #ifdef CPU_DRAND48
                                    random = cpu_drand48_generate(*(thread_instance->thread_random_state));
                                #endif

                                machine_a = (int)floor(random * thread_instance->etc->machines_count);

                               // fprintf(stdout, "[DEBUG] Random machine_a = %d\n", machine_a);
                            } else {
                                machine_a = get_worst_ct_machine_id(selected_solution);

                                //fprintf(stdout, "[DEBUG] Worst CT machine_a = %d\n", machine_a);
                            }

                            #ifdef CPU_MERSENNE_TWISTER
                                random = cpu_mt_generate(*(thread_instance->thread_random_state));
                            #endif
                            #ifdef CPU_RAND
                                random = cpu_rand_generate(*(thread_instance->thread_random_state));
                            #endif
                            #ifdef CPU_DRAND48
                                random = cpu_drand48_generate(*(thread_instance->thread_random_state));
                            #endif

                            if (random > PALS_CPU_1POP_SEARCH__MAKESPAN_GREEDY_PSEL_BEST) {
                                #ifdef CPU_MERSENNE_TWISTER
                                    random = cpu_mt_generate(*(thread_instance->thread_random_state));
                                #endif
                                #ifdef CPU_RAND
                                    random = cpu_rand_generate(*(thread_instance->thread_random_state));
                                #endif
                                #ifdef CPU_DRAND48
                                    random = cpu_drand48_generate(*(thread_instance->thread_random_state));
                                #endif

                                // Siempre selecciono la segunda mquina aleatoriamente.
                                machine_b = (int)floor(random * (thread_instance->etc->machines_count - 1));

                                //fprintf(stdout, "[DEBUG] Random machine_b = %d\n", machine_b);
                            } else {
                                machine_b = get_best_ct_machine_id(selected_solution);

                                //fprintf(stdout, "[DEBUG] Best CT machine_b = %d\n", machine_b);
                            }
                        }
                        else if (search_type == PALS_CPU_1POP_SEARCH__ENERGY_GREEDY)
                        {
                            //fprintf(stdout, "[DEBUG] Energy greedy\n");

                            #ifdef CPU_MERSENNE_TWISTER
                                random = cpu_mt_generate(*(thread_instance->thread_random_state));
                            #endif
                            #ifdef CPU_RAND
                                random = cpu_rand_generate(*(thread_instance->thread_random_state));
                            #endif
                            #ifdef CPU_DRAND48
                                random = cpu_drand48_generate(*(thread_instance->thread_random_state));
                            #endif

                            if (random > PALS_CPU_1POP_SEARCH__ENERGY_GREEDY_PSEL_WORST) {
                                #ifdef CPU_MERSENNE_TWISTER
                                    random = cpu_mt_generate(*(thread_instance->thread_random_state));
                                #endif
                                #ifdef CPU_RAND
                                    random = cpu_rand_generate(*(thread_instance->thread_random_state));
                                #endif
                                #ifdef CPU_DRAND48
                                    random = cpu_drand48_generate(*(thread_instance->thread_random_state));
                                #endif

                                machine_a = (int)floor(random * thread_instance->etc->machines_count);

                                //fprintf(stdout, "[DEBUG] Random machine_a = %d\n", machine_a);
                            } else {
                                machine_a = get_worst_energy_machine_id(selected_solution);

                                //fprintf(stdout, "[DEBUG] Worst energy machine_a = %d\n", machine_a);
                            }

                            #ifdef CPU_MERSENNE_TWISTER
                                random = cpu_mt_generate(*(thread_instance->thread_random_state));
                            #endif
                            #ifdef CPU_RAND
                                random = cpu_rand_generate(*(thread_instance->thread_random_state));
                            #endif
                            #ifdef CPU_DRAND48
                                random = cpu_drand48_generate(*(thread_instance->thread_random_state));
                            #endif

                            if (random > PALS_CPU_1POP_SEARCH__ENERGY_GREEDY_PSEL_BEST) {
                                #ifdef CPU_MERSENNE_TWISTER
                                    random = cpu_mt_generate(*(thread_instance->thread_random_state));
                                #endif
                                #ifdef CPU_RAND
                                    random = cpu_rand_generate(*(thread_instance->thread_random_state));
                                #endif
                                #ifdef CPU_DRAND48
                                    random = cpu_drand48_generate(*(thread_instance->thread_random_state));
                                #endif

                                // Siempre selecciono la segunda mquina aleatoriamente.
                                machine_b = (int)floor(random * (thread_instance->etc->machines_count - 1));

                                //fprintf(stdout, "[DEBUG] Random machine_b = %d\n", machine_b);
                            } else {
                                machine_b = get_best_energy_machine_id(selected_solution);

                                //fprintf(stdout, "[DEBUG] Worst energy machine_b = %d\n", machine_b);
                            }
                        }
                        else
                        {
                            #ifdef CPU_MERSENNE_TWISTER
                                random = cpu_mt_generate(*(thread_instance->thread_random_state));
                            #endif
                            #ifdef CPU_RAND
                                random = cpu_rand_generate(*(thread_instance->thread_random_state));
                            #endif
                            #ifdef CPU_DRAND48
                                random = cpu_drand48_generate(*(thread_instance->thread_random_state));
                            #endif

                            // La estrategia es aleatoria.
                            machine_a = (int)floor(random * thread_instance->etc->machines_count);

                            #ifdef CPU_MERSENNE_TWISTER
                                random = cpu_mt_generate(*(thread_instance->thread_random_state));
                            #endif
                            #ifdef CPU_RAND
                                random = cpu_rand_generate(*(thread_instance->thread_random_state));
                            #endif
                            #ifdef CPU_DRAND48
                                random = cpu_drand48_generate(*(thread_instance->thread_random_state));
                            #endif

                            // Siempre selecciono la segunda mquina aleatoriamente.
                            machine_b = (int)floor(random * (thread_instance->etc->machines_count - 1));
                        }

                        machine_a_task_count = get_machine_tasks_count(selected_solution, machine_a);
                        while (machine_a_task_count == 0) {
                            machine_a = (machine_a + 1) % thread_instance->etc->machines_count;
                            machine_a_task_count = get_machine_tasks_count(selected_solution, machine_a);
                        }

                        if (machine_a == machine_b) machine_b = (machine_b + 1) % thread_instance->etc->machines_count;
                        machine_b_task_count = get_machine_tasks_count(selected_solution, machine_b);

                        #ifdef CPU_MERSENNE_TWISTER
                            random = cpu_mt_generate(*(thread_instance->thread_random_state));
                        #endif
                        #ifdef CPU_RAND
                            random = cpu_rand_generate(*(thread_instance->thread_random_state));
                        #endif
                        #ifdef CPU_DRAND48
                            random = cpu_drand48_generate(*(thread_instance->thread_random_state));
                        #endif

                        task_x = (int)floor(random * machine_a_task_count);

                        machine_a_energy_idle = get_energy_idle_value(thread_instance->energy, machine_a);
                        machine_a_energy_max = get_energy_max_value(thread_instance->energy, machine_a);
                        machine_b_energy_idle = get_energy_idle_value(thread_instance->energy, machine_b);
                        machine_b_energy_max = get_energy_max_value(thread_instance->energy, machine_b);

                        current_makespan = get_makespan(selected_solution);

                        #ifdef CPU_MERSENNE_TWISTER
                        random = cpu_mt_generate(*(thread_instance->thread_random_state));
                        #endif
                        #ifdef CPU_RAND
                        random = cpu_rand_generate(*(thread_instance->thread_random_state));
                        #endif
                        #ifdef CPU_DRAND48
                        random = cpu_drand48_generate(*(thread_instance->thread_random_state));
                        #endif

                        top_task_a = (int)floor(random * PALS_CPU_1POP_WORK__SRC_TASK_NHOOD) + 1;
                        if (top_task_a > machine_a_task_count) top_task_a = machine_a_task_count;

                        #ifdef CPU_MERSENNE_TWISTER
                        random = cpu_mt_generate(*(thread_instance->thread_random_state));
                        #endif
                        #ifdef CPU_RAND
                        random = cpu_rand_generate(*(thread_instance->thread_random_state));
                        #endif
                        #ifdef CPU_DRAND48
                        random = cpu_drand48_generate(*(thread_instance->thread_random_state));
                        #endif

                        top_task_b = (int)floor(random * PALS_CPU_1POP_WORK__DST_TASK_NHOOD) + 1;
                        if (top_task_b > machine_b_task_count) top_task_b = machine_b_task_count;

                        #ifdef CPU_MERSENNE_TWISTER
                        random = cpu_mt_generate(*(thread_instance->thread_random_state));
                        #endif
                        #ifdef CPU_RAND
                        random = cpu_rand_generate(*(thread_instance->thread_random_state));
                        #endif
                        #ifdef CPU_DRAND48
                        random = cpu_drand48_generate(*(thread_instance->thread_random_state));
                        #endif

                        top_machine_b = (int)floor(random * PALS_CPU_1POP_WORK__DST_MACH_NHOOD) + 1;
                        if (top_machine_b > thread_instance->etc->machines_count) top_machine_b = thread_instance->etc->machines_count;

                        #ifdef CPU_MERSENNE_TWISTER
                        random = cpu_mt_generate(*(thread_instance->thread_random_state));
                        #endif
                        #ifdef CPU_RAND
                        random = cpu_rand_generate(*(thread_instance->thread_random_state));
                        #endif
                        #ifdef CPU_DRAND48
                        random = cpu_drand48_generate(*(thread_instance->thread_random_state));
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
                            random = cpu_mt_generate(*(thread_instance->thread_random_state));
                            #endif
                            #ifdef CPU_RAND
                            random = cpu_rand_generate(*(thread_instance->thread_random_state));
                            #endif
                            #ifdef CPU_DRAND48
                            random = cpu_drand48_generate(*(thread_instance->thread_random_state));
                            #endif

                            mov_type = PALS_CPU_1POP_SEARCH_OP__SWAP;
                            if ((random < PALS_CPU_1POP_SEARCH_OP_BALANCE__SWAP) && (machine_b_task_count > 0))
                            {
                                mov_type = PALS_CPU_1POP_SEARCH_OP__SWAP;
                            }
                            else //if (random < PALS_CPU_1POP_SEARCH_OP_BALANCE__SWAP + PALS_CPU_1POP_SEARCH_OP_BALANCE__MOVE)
                            {
                                mov_type = PALS_CPU_1POP_SEARCH_OP__MOVE;
                            }

                            if (mov_type == PALS_CPU_1POP_SEARCH_OP__SWAP)
                            {
                                for (task_y_offset = 0; (task_y_offset < top_task_b); task_y_offset++)
                                {
                                    task_y_pos = (task_y + task_y_offset) % machine_b_task_count;
                                    task_y_current = get_machine_task_id(selected_solution, machine_b, task_y_pos);

                                    // Mquina 1.
                                    machine_a_ct_old = get_machine_compute_time(selected_solution, machine_a);

                                    machine_a_ct_new = machine_a_ct_old -
                                        get_etc_value(thread_instance->etc, machine_a, task_x_current) +
                                        get_etc_value(thread_instance->etc, machine_a, task_y_current);

                                    // Mquina 2.
                                    machine_b_ct_old = get_machine_compute_time(selected_solution, machine_b);

                                    machine_b_ct_new = machine_b_ct_old -
                                        get_etc_value(thread_instance->etc, machine_b, task_y_current) +
                                        get_etc_value(thread_instance->etc, machine_b, task_x_current);

                                    #ifdef CPU_MERSENNE_TWISTER
                                    random = cpu_mt_generate(*(thread_instance->thread_random_state));
                                    #endif
                                    #ifdef CPU_RAND
                                    random = cpu_rand_generate(*(thread_instance->thread_random_state));
                                    #endif
                                    #ifdef CPU_DRAND48
                                    random = cpu_drand48_generate(*(thread_instance->thread_random_state));
                                    #endif

                                    if ((search_type == PALS_CPU_1POP_SEARCH__MAKESPAN_GREEDY) ||
                                        ((random < 0.5) && (search_type == PALS_CPU_1POP_SEARCH__RANDOM_GREEDY)))
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

                                    if ((search_type == PALS_CPU_1POP_SEARCH__ENERGY_GREEDY) ||
                                        ((random >= 0.5) && (search_type == PALS_CPU_1POP_SEARCH__RANDOM_GREEDY)))
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
                            else if (mov_type == PALS_CPU_1POP_SEARCH_OP__MOVE)
                            {
                                machine_b_current = machine_b;

                                for (machine_b_offset = 0; (machine_b_offset < top_machine_b); machine_b_offset++)
                                {
                                    if (machine_b_offset == 1) {
                                        #ifdef CPU_MERSENNE_TWISTER
                                        random = cpu_mt_generate(*(thread_instance->thread_random_state));
                                        #endif
                                        #ifdef CPU_RAND
                                        random = cpu_rand_generate(*(thread_instance->thread_random_state));
                                        #endif
                                        #ifdef CPU_DRAND48
                                        random = cpu_drand48_generate(*(thread_instance->thread_random_state));
                                        #endif

                                        // Siempre selecciono la segunda mquina aleatoriamente.
                                        machine_b_current = (int)floor(random * (thread_instance->etc->machines_count - 1));

                                        if (machine_b_current == machine_a) machine_b_current = (machine_b_current + 1) % thread_instance->etc->machines_count;
                                    } 
                                    else if (machine_b_offset > 1) {
                                        if (machine_b + machine_b_offset != machine_a) {
                                            machine_b_current = (machine_b + machine_b_offset) % thread_instance->etc->machines_count;
                                        } else {
                                            machine_b_current = (machine_b + machine_b_offset + 1) % thread_instance->etc->machines_count;
                                        }
                                    }

                                    if (machine_b_current != machine_a)
                                    {
                                        machine_b_current_energy_idle = get_energy_idle_value(thread_instance->energy, machine_b_current);
                                        machine_b_current_energy_max = get_energy_max_value(thread_instance->energy, machine_b_current);

                                        // Mquina 1.
                                        machine_a_ct_old = get_machine_compute_time(selected_solution, machine_a);
                                        machine_a_ct_new = machine_a_ct_old - get_etc_value(thread_instance->etc, machine_a, task_x_current);

                                        // Mquina 2.
                                        machine_b_ct_old = get_machine_compute_time(selected_solution, machine_b_current);
                                        machine_b_ct_new = machine_b_ct_old + get_etc_value(thread_instance->etc, machine_b_current, task_x_current);

                                        #ifdef CPU_MERSENNE_TWISTER
                                        random = cpu_mt_generate(*(thread_instance->thread_random_state));
                                        #endif
                                        #ifdef CPU_RAND
                                        random = cpu_rand_generate(*(thread_instance->thread_random_state));
                                        #endif
                                        #ifdef CPU_DRAND48
                                        random = cpu_drand48_generate(*(thread_instance->thread_random_state));
                                        #endif

                                        if ((search_type == PALS_CPU_1POP_SEARCH__MAKESPAN_GREEDY) ||
                                            ((random < 0.5) && (search_type == PALS_CPU_1POP_SEARCH__RANDOM_GREEDY)))
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

                                        if ((search_type == PALS_CPU_1POP_SEARCH__ENERGY_GREEDY) ||
                                            ((random >= 0.5) && (search_type == PALS_CPU_1POP_SEARCH__RANDOM_GREEDY)))
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

                            thread_instance->total_swaps++;
                            //printf("Makespan %f Energy %f\n", get_makespan(selected_solution), get_energy(selected_solution));
                        }
                        else if ((task_x_best_move_pos != -1) && (machine_b_best_move_id != -1))
                        {
                            // Muevo la tarea!
                            #if defined(DEBUG_DEV) 
                            fprintf(stdout, "[DEBUG] Ejecuto un MOVE! %f %f (%d, %d, %d)\n",
                                best_delta_makespan, best_delta_energy, machine_a, task_x_best_move_pos, machine_b_best_move_id);
                            #endif
                            
                            move_task_to_machine_by_pos(selected_solution, machine_a, task_x_best_move_pos, machine_b_best_move_id);

                            thread_instance->total_moves++;
                            //printf("Makespan %f Energy %f\n", get_makespan(selected_solution), get_energy(selected_solution));
                        }

                        #if defined(DEBUG_DEV) 
                        validate_solution(selected_solution);
                    
                        if ((current_makespan < get_makespan(selected_solution)) && (current_energy < get_energy(selected_solution))) {
                            refresh_energy(selected_solution);
                            refresh_makespan(selected_solution);

                            fprintf(stdout, "[ERROR] EMPEORA!\n");
                            fprintf(stdout, "[ERROR] Makespan %f ahora %f\n", current_makespan, get_makespan(selected_solution));
                            fprintf(stdout, "[ERROR] Energy   %f ahora %f\n", current_energy, get_energy(selected_solution));

                            exit(-1);
                        }
                        #endif
                    }                // Termino el loop con la iteracin del thread

                    if ((original_makespan > get_makespan(selected_solution)) ||
                        (original_energy > get_energy(selected_solution)))
                    {
                        new_solution_eval = 0;

                        // Lo mejore. Intento obtener lock de la poblacion.
                        mutex_locked = pthread_mutex_trylock(thread_instance->population_mutex);

                        if (mutex_locked == 0) {
                            // Chequeo si la nueva solucion es no-dominada.
                            
                            #if defined(ARCHIVER_AGA)
                                new_solution_eval = archivers_aga(thread_instance, selected_solution_pos);
                            #else
                                new_solution_eval = pals_cpu_1pop_adhoc_arch(thread_instance, selected_solution_pos);
                            #endif

                            pthread_mutex_unlock(thread_instance->population_mutex);

                            if (new_solution_eval == 1)
                            {
                                thread_instance->ts_last_found = ts_current;
                                thread_instance->iter_last_found = thread_instance->total_iterations;

                                if (search_type == PALS_CPU_1POP_SEARCH__MAKESPAN_GREEDY)
                                {
                                    thread_instance->total_success_makespan_greedy_searches++;
                                }
                                else if (search_type == PALS_CPU_1POP_SEARCH__ENERGY_GREEDY)
                                {
                                    thread_instance->total_success_energy_greedy_searches++;
                                }
                                else
                                {
                                    thread_instance->total_success_random_greedy_searches++;
                                }
                            } else {
                                thread_instance->total_soluciones_evolucionadas_dominadas++;
                            }

                            #if defined(DEBUG_DEV)
                            fprintf(stdout, "[DEBUG] Cantidad de individuos en la poblacion: %d\n", *(thread_instance->population_count));
                            validate_thread_instance(thread_instance);
                            #endif

                            // Libero la posicion seleccionada.
                            selected_solution_pos = -1;
                        } else {
                            // Algun otro thread esta trabajando sobre la población.
                            // Intento hacer otro loop de trabajo y vuelvo a probar.
                            #if defined(DEBUG_DEV) 
                            fprintf(stdout, "[DEBUG] Re do iteration!\n");
                            #endif

                            #ifdef CPU_MERSENNE_TWISTER
                            random = cpu_mt_generate(*(thread_instance->thread_random_state));
                            #endif
                            #ifdef CPU_RAND
                            random = cpu_rand_generate(*(thread_instance->thread_random_state));
                            #endif
                            #ifdef CPU_DRAND48
                            random = cpu_drand48_generate(*(thread_instance->thread_random_state));
                            #endif

                            work_do_iteration = 1;
                            work_iteration_size = (int)floor(random * PALS_CPU_1POP_WORK__THREAD_ITERATIONS / PALS_CPU_1POP_WORK__THREAD_RE_WORK_FACTOR);

                            thread_instance->total_re_iterations++;
                        }
                    }
                    else
                    {
                        // No lo pude mejorar.
                        thread_instance->total_soluciones_no_evolucionadas++;
                    }
                }
            }
        }

        clock_gettime(CLOCK_REALTIME, &ts_current);
    }

    if (selected_solution_pos != -1) {
        if (thread_instance->population[selected_solution_pos].status == SOLUTION__STATUS_NOT_READY) {
            thread_instance->population[selected_solution_pos].status = SOLUTION__STATUS_EMPTY;
        }
    }

    #if defined(DEBUG_DEV) 
    fprintf(stdout, "[DEBUG] Me mandaron a terminar o se acabo el tiempo! Tengo algo para hacer?\n");
    #endif

    return NULL;
}
