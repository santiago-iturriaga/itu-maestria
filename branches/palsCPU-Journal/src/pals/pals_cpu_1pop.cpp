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
#include "../random/cpu_rand.h"
#include "../random/cpu_drand48.h"
#include "../random/cpu_mt.h"

#include "archivers/adhoc.h"
#include "archivers/aga.h"
#include "ls_selection/evol_guide_simple.h"
#include "ls_selection/evol_guide_complex.h"
#include "ls_selection/machine_sel_simple.h"
#include "ls_selection/machine_sel_complex.h"

#include "pals_cpu_1pop.h"

inline void rand_generate(pals_cpu_1pop_thread_arg *thread_instance, double &random) {
    #ifdef CPU_MERSENNE_TWISTER
    random = cpu_mt_generate(*(thread_instance->thread_random_state));
    #endif
    #ifdef CPU_RAND
    random = cpu_rand_generate(*(thread_instance->thread_random_state));
    #endif
    #ifdef CPU_DRAND48
    random = cpu_drand48_generate(*(thread_instance->thread_random_state));
    #endif
}

void pals_cpu_1pop(struct params &input, struct etc_matrix *etc, struct energy_matrix *energy)
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

    // Bloqueo la ejecucion hasta que terminen todos los hilos.
    for(int i = 0; i < instance.count_threads; i++)
    {
        if(pthread_join(instance.threads[i], NULL))
        {
            printf("Could not join thread %d\n", i);
            exit(EXIT_FAILURE);
        }
        else
        {
            if (DEBUG) printf("[DEBUG] thread %d <OK>\n", i);
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

        if (DEBUG_DEV)
        {
            for (int i = 0; i < instance.population_max_size; i++)
            {
                if (instance.population[i].status == SOLUTION__STATUS_READY)
                {
                    validate_solution(&(instance.population[i]));
                }
            }
        }
    }
    // <=========== DEBUG

    if (DEBUG)
    {
        fprintf(stdout, "== Population =================================================\n");
        for (int i = 0; i < instance.population_max_size; i++)
        {
            if (instance.population[i].status > SOLUTION__STATUS_EMPTY)
            {
                fprintf(stdout, "Solucion %d: %f %f\n", i, get_makespan(&(instance.population[i])), get_energy(&(instance.population[i])));
            }
        }
    }
    else
    {
        if (!OUTPUT_SOLUTION)
        {
            fprintf(stdout, "== Population =================================================\n");
            for (int i = 0; i < instance.population_max_size; i++)
            {
                /*if (instance.population[i].status > SOLUTION__STATUS_EMPTY)
                {*/
                    fprintf(stdout, "%f %f (%d)\n", get_makespan(&(instance.population[i])), get_energy(&(instance.population[i])), 
                        instance.population[i].status);
                /*}*/
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
    }

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

    #ifdef ARCHIVER_ADHOC
    // No es necesario...
    #endif
    #ifdef ARCHIVER_AGA
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

        #ifdef ARCHIVER_ADHOC
        // No es necesario...
        #endif
        #ifdef ARCHIVER_AGA
        empty_instance.threads_args[i].archiver_state = empty_instance.archiver_state;
        #endif

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
    
    #ifdef ARCHIVER_ADHOC
    // No es necesario...
    #endif
    #ifdef ARCHIVER_AGA
    archivers_aga_free(&instance);
    #endif

    pthread_mutex_destroy(&(instance.population_mutex));
    pthread_barrier_destroy(&(instance.sync_barrier));
}

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

    while ((terminate == 0) &&
        (ts_current.tv_sec - thread_instance->ts_start.tv_sec < thread_instance->max_time_secs) &&
        (thread_instance->total_iterations < thread_instance->max_iterations) &&
        (thread_instance->global_total_iterations[0] < thread_instance->max_iterations))
    {
        work_type = *(thread_instance->work_type);
        if (DEBUG_DEV) printf("[DEBUG] [THREAD=%d] Work type = %d\n", thread_instance->thread_idx, work_type);

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
                pthread_mutex_lock(thread_instance->population_mutex);

                thread_instance->population[thread_instance->thread_idx].status = SOLUTION__STATUS_NOT_READY;

                pthread_mutex_unlock(thread_instance->population_mutex);

                // Inicializo el individuo que me toca.
                init_empty_solution(thread_instance->etc, thread_instance->energy, &(thread_instance->population[thread_instance->thread_idx]));

                double random;
                rand_generate(thread_instance, random);

                int random_task = (int)floor(random * thread_instance->etc->tasks_count);
                
                #ifdef INIT_MCT
                compute_custom_mct(&(thread_instance->population[thread_instance->thread_idx]), random_task);
                #endif
                #ifdef INIT_MINMIN
                compute_minmin(&(thread_instance->population[thread_instance->thread_idx]));
                #endif

                pthread_mutex_lock(thread_instance->population_mutex);

                #ifndef ARCHVIER_ADHOC
                archivers_adhoc(thread_instance, thread_instance->thread_idx);
                #endif
                #ifndef ARCHIVER_AGA
                archivers_aga(thread_instance, thread_instance->thread_idx);
                #endif

                if (DEBUG_DEV)
                {
                    fprintf(stdout, "[DEBUG] Population\n");
                    fprintf(stdout, "        Population_count: %d\n", *(thread_instance->population_count));

                    for (int i = 0; i < thread_instance->population_max_size; i++)
                    {
                        fprintf(stdout, " >> sol.pos[%d] init=%d status=%d\n", i,
                            thread_instance->population[i].initialized,
                            thread_instance->population[i].status);
                    }
                }
                
                // !!!!!!!!!!!!!!!!!!!
                //fprintf(stdout, "[INIT] makespan(%f) energy(%f)\n", get_makespan(&(thread_instance->population[thread_instance->thread_idx])),
                //    get_energy(&(thread_instance->population[thread_instance->thread_idx])));

                pthread_mutex_unlock(thread_instance->population_mutex);

                if (DEBUG)
                {
                    fprintf(stdout, "[DEBUG] Initializing individual %d (%f %f)\n",
                        thread_instance->thread_idx, get_makespan(&(thread_instance->population[thread_instance->thread_idx])),
                        get_energy(&(thread_instance->population[thread_instance->thread_idx])));
                }

                // Timming -----------------------------------------------------
                timming_end(">> Random MCT Time", ts_mct);
                // Timming -----------------------------------------------------

                if (DEBUG_DEV) validate_solution(&(thread_instance->population[thread_instance->thread_idx]));
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
            double random = 0.0; // Variable random multi-proposito :)

            // Busco un lugar libre en la poblacin para clonar un individuo y evolucionarlo ==================
            if (selected_solution_pos == -1)
            {
                pthread_mutex_lock(thread_instance->population_mutex);

                int candidate_to_del_pos = -1;
                
                for (int i = 0; (i < thread_instance->population_max_size) && (selected_solution_pos == -1); i++)
                {
                    if (thread_instance->population[i].status == SOLUTION__STATUS_EMPTY)
                    {
                        thread_instance->population[i].status = SOLUTION__STATUS_NOT_READY;
                        selected_solution_pos = i;

                        if (DEBUG_DEV) fprintf(stdout, "[DEBUG] Found individual %d free\n", selected_solution_pos);
                        
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
                // No se que hacer... PANICO! PANICO!... listo termino...
                fprintf(stdout, "[ERROR] Hilo finalizado! PANIC!!!\n");

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
                    if (DEBUG_DEV) fprintf(stdout, "[DEBUG] Initializing individual %d\n", selected_solution_pos);
                    init_empty_solution(thread_instance->etc, thread_instance->energy, selected_solution);
                }

                // Sorteo la solucion con la que me toca trabajar  =====================================================
                
                rand_generate(thread_instance, random);

                pthread_mutex_lock(thread_instance->population_mutex);
                thread_instance->global_total_iterations[0] += local_iteration_count;
                local_iteration_count = 0;

                int random_sol_index = (int)floor(random * (*(thread_instance->population_count)));

                if (DEBUG_DEV)
                {
                    fprintf(stdout, "[DEBUG] Random selection\n");
                    fprintf(stdout, "        Population_count: %d\n", *(thread_instance->population_count));
                    fprintf(stdout, "        Random          : %f\n", random);
                    fprintf(stdout, "        Random_sol_index: %d\n", random_sol_index);

                    for (int i = 0; i < thread_instance->population_max_size; i++)
                    {
                        fprintf(stdout, " >> sol.pos[%d] init=%d status=%d\n", i,
                            thread_instance->population[i].initialized,
                            thread_instance->population[i].status);
                    }
                }

                int current_sol_pos = -1;
                int current_sol_index = -1;

                for (int i = 0; (i < thread_instance->population_max_size) && (current_sol_pos == -1); i++)
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
                if (DEBUG_DEV) fprintf(stdout, "[DEBUG] Cloning individual %d to %d\n", current_sol_pos, selected_solution_pos);
                clone_solution(selected_solution, &(thread_instance->population[current_sol_pos]), 0);

                pthread_mutex_unlock(thread_instance->population_mutex);

                // Determino la estrategia de busqueda del hilo  =====================================================
                if (DEBUG_DEV)
                {
                    fprintf(stdout, "[DEBUG] Selected individual\n");
                    fprintf(stdout, "        Original_solutiol_pos = %d\n", current_sol_pos);
                    fprintf(stdout, "        Selected_solution_pos = %d\n", selected_solution_pos);
                    fprintf(stdout, "        Selected_solution.status = %d\n", selected_solution->status);
                    fprintf(stdout, "        Selected_solution.initializd = %d\n", selected_solution->initialized);
                }

                float original_makespan = get_makespan(selected_solution);
                float original_energy = get_energy(selected_solution);

                rand_generate(thread_instance, random);

                int search_type;
                double search_type_random = 0.0;

                rand_generate(thread_instance, search_type_random);

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

                int work_do_iteration = 1;

                /*int work_iteration_size = (int)floor((PALS_CPU_1POP_WORK__THREAD_ITERATIONS / PALS_CPU_1POP_WORK__THREAD_RE_WORK_FACTOR) +
                    (random * (PALS_CPU_1POP_WORK__THREAD_ITERATIONS - (PALS_CPU_1POP_WORK__THREAD_ITERATIONS / PALS_CPU_1POP_WORK__THREAD_RE_WORK_FACTOR))));*/

                int work_iteration_size = (int)floor(random * PALS_CPU_1POP_WORK__THREAD_ITERATIONS) + 1;

                while (work_do_iteration == 1)
                {
                    work_do_iteration = 0;

                    for (int search_iteration = 0; search_iteration < work_iteration_size; search_iteration++)
                    {
                        thread_instance->total_iterations++;
                        local_iteration_count++;

                        // Determino las maquinas de inicio para la busqueda.
                        int machine_a, machine_b;
                        
                        #ifdef MACH_SEL_SIMPLE
                        machines_simple_selection(thread_instance, selected_solution, search_type, machine_a, machine_b);
                        #endif
                        #ifdef MACH_SEL_COMPLEX
                        machines_complex_selection(thread_instance, selected_solution, search_type, machine_a, machine_b);
                        #endif

                        int machine_a_task_count = get_machine_tasks_count(selected_solution, machine_a);
                        int machine_b_task_count = get_machine_tasks_count(selected_solution, machine_b);

                        rand_generate(thread_instance, random);

                        int task_x;
                        task_x = (int)floor(random * machine_a_task_count);

                        int task_x_pos;
                        int task_x_current;

                        rand_generate(thread_instance, random);

                        int top_task_a = (int)floor(random * PALS_CPU_1POP_WORK__SRC_TASK_NHOOD) + 1;
                        if (top_task_a > machine_a_task_count) top_task_a = machine_a_task_count;

                        rand_generate(thread_instance, random);

                        int top_task_b = (int)floor(random * PALS_CPU_1POP_WORK__DST_TASK_NHOOD) + 1;
                        if (top_task_b > machine_b_task_count) top_task_b = machine_b_task_count;

                        rand_generate(thread_instance, random);

                        int top_machine_b = (int)floor(random * PALS_CPU_1POP_WORK__DST_MACH_NHOOD) + 1;
                        if (top_machine_b > thread_instance->etc->machines_count) top_machine_b = thread_instance->etc->machines_count;

                        rand_generate(thread_instance, random);

                        int task_y = (int)floor(random * machine_b_task_count);

                        float current_makespan = get_makespan(selected_solution);
                        float current_energy = get_energy(selected_solution);

                        float best_delta_makespan;
                        best_delta_makespan = current_makespan;

                        float best_delta_energy;
                        best_delta_energy = 0.0;

                        int task_x_best_move_pos;
                        task_x_best_move_pos = -1;

                        int machine_b_best_move_id;
                        machine_b_best_move_id = -1;

                        int task_x_best_swap_pos;
                        task_x_best_swap_pos = -1;

                        int task_y_best_swap_pos;
                        task_y_best_swap_pos = -1;

                        for (int task_x_offset = 0; (task_x_offset < top_task_a); task_x_offset++)
                        {
                            task_x_pos = (task_x + task_x_offset) % machine_a_task_count;
                            task_x_current = get_machine_task_id(selected_solution, machine_a, task_x_pos);

                            // Determino que tipo movimiento va a realizar el hilo.
                            rand_generate(thread_instance, random);

                            int mov_type = PALS_CPU_1POP_SEARCH_OP__SWAP;
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
                                int task_y_pos, task_y_current;
                                for (int task_y_offset = 0; (task_y_offset < top_task_b); task_y_offset++)
                                {
                                    task_y_pos = (task_y + task_y_offset) % machine_b_task_count;
                                    task_y_current = get_machine_task_id(selected_solution, machine_b, task_y_pos);

                                    #ifdef EVOL_GUIDE_SIMPLE
                                    ls_best_swap_simple_selection(thread_instance, selected_solution,
                                        search_type, machine_a, machine_b, task_x_pos, task_x_current,
                                        task_y_pos, task_y_current, best_delta_makespan, best_delta_energy,
                                        task_x_best_move_pos, machine_b_best_move_id, task_x_best_swap_pos,
                                        task_y_best_swap_pos);
                                    #endif
                                    #ifdef EVOL_GUIDE_COMPLEX
                                    ls_best_swap_complex_selection(thread_instance, selected_solution,
                                        search_type, machine_a, machine_b, task_x_pos, task_x_current,
                                        task_y_pos, task_y_current, best_delta_makespan, best_delta_energy,
                                        task_x_best_move_pos, machine_b_best_move_id, task_x_best_swap_pos,
                                        task_y_best_swap_pos);
                                    #endif
                                }
                            }
                            else if (mov_type == PALS_CPU_1POP_SEARCH_OP__MOVE)
                            {
                                int machine_b_current = machine_b;

                                for (int machine_b_offset = 0; (machine_b_offset < top_machine_b); machine_b_offset++)
                                {
                                    if (machine_b_offset == 1)
                                    {
                                        rand_generate(thread_instance, random);

                                        // Siempre selecciono la segunda mquina aleatoriamente.
                                        machine_b_current = (int)floor(random * (thread_instance->etc->machines_count - 1));

                                        if (machine_b_current == machine_a) machine_b_current = (machine_b_current + 1) % thread_instance->etc->machines_count;
                                    }
                                    else if (machine_b_offset > 1)
                                    {
                                        if (machine_b + machine_b_offset != machine_a)
                                        {
                                            machine_b_current = (machine_b + machine_b_offset) % thread_instance->etc->machines_count;
                                        }
                                        else
                                        {
                                            machine_b_current = (machine_b + machine_b_offset + 1) % thread_instance->etc->machines_count;
                                        }
                                    }

                                    if (machine_b_current != machine_a)
                                    {
                                        #ifdef EVOL_GUIDE_SIMPLE
                                        ls_best_move_simple_selection(thread_instance, selected_solution,
                                            search_type, machine_a, machine_b_current, task_x_pos, task_x_current,
                                            best_delta_makespan, best_delta_energy, task_x_best_move_pos,
                                            machine_b_best_move_id, task_x_best_swap_pos, task_y_best_swap_pos);
                                        #endif
                                        #ifdef EVOL_GUIDE_COMPLEX
                                        ls_best_move_complex_selection(thread_instance, selected_solution,
                                            search_type, machine_a, machine_b_current, task_x_pos, task_x_current,
                                            best_delta_makespan, best_delta_energy, task_x_best_move_pos,
                                            machine_b_best_move_id, task_x_best_swap_pos, task_y_best_swap_pos);
                                        #endif
                                        
                                        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                        // refresh(selected_solution);
                                    }
                                }
                            }    // Termino el IF de SWAP/MOVE
                        }        // Termino el loop de TASK_A

                        // Hago los cambios ======================================================================================
                        if ((task_x_best_swap_pos != -1) && (task_y_best_swap_pos != -1))
                        {
                            // Intercambio las tareas!
                            if (DEBUG_DEV) fprintf(stdout, "[DEBUG] Ejecuto un SWAP! %f %f (%d, %d, %d, %d)\n",
                                    best_delta_makespan, best_delta_energy, machine_a, task_x_best_swap_pos, machine_b, task_y_best_swap_pos);
                          
                            swap_tasks_by_pos(selected_solution, machine_a, task_x_best_swap_pos, machine_b, task_y_best_swap_pos);

                            thread_instance->total_swaps++;
                        }
                        else if ((task_x_best_move_pos != -1) && (machine_b_best_move_id != -1))
                        {
                            // Muevo la tarea!
                            if (DEBUG_DEV) fprintf(stdout, "[DEBUG] Ejecuto un MOVE! %f %f (%d, %d, %d)\n",
                                    best_delta_makespan, best_delta_energy, machine_a, task_x_best_move_pos, machine_b_best_move_id);
                            
                            move_task_to_machine_by_pos(selected_solution, machine_a, task_x_best_move_pos, machine_b_best_move_id);

                            thread_instance->total_moves++;
                        }
                    } // Termino el loop con la iteracion del thread

                    refresh(selected_solution);

                    if ((original_makespan > get_makespan(selected_solution)) ||
                        (original_energy > get_energy(selected_solution)))
                    {
                        refresh_energy(selected_solution);

                        int mutex_locked;
                        int new_solution_eval = 0;

                        // Lo mejore. Intento obtener lock de la poblacion.
                        mutex_locked = pthread_mutex_trylock(thread_instance->population_mutex);

                        if (mutex_locked == 0)
                        {
                            // Chequeo si la nueva solucion es no-dominada.
                            #ifdef ARCHIVER_ADHOC
                            new_solution_eval = archivers_adhoc(thread_instance, selected_solution_pos);
                            #endif
                            #ifdef ARCHIVER_AGA
                            new_solution_eval = archivers_aga(thread_instance, selected_solution_pos);
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
                            }
                            else
                            {
                                thread_instance->total_soluciones_evolucionadas_dominadas++;
                            }

                            if (DEBUG_DEV)
                            {
                                fprintf(stdout, "[DEBUG] Cantidad de individuos en la poblacion: %d\n", *(thread_instance->population_count));
                                validate_thread_instance(thread_instance);
                            }

                            // Libero la posicion seleccionada.
                            selected_solution_pos = -1;
                        }
                        else
                        {
                            // Algun otro thread esta trabajando sobre la poblacion.
                            // Intento hacer otro loop de trabajo y vuelvo a probar.
                            if (DEBUG_DEV) fprintf(stdout, "[DEBUG] Re do iteration!\n");

                            //rand_generate(thread_instance, random);
                            random = 1;

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

    if (DEBUG_DEV) fprintf(stdout, "[DEBUG] Me mandaron a terminar o se acabo el tiempo! Tengo algo para hacer?\n");

    return NULL;
}
