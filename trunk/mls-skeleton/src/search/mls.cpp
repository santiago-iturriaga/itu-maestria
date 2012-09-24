#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <assert.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <semaphore.h>

#include "mls.h"

#include "../config.h"
#include "../random/cpu_mt.h"
#include "archivers/aga.h"

void mls()
{
    int seed = 0;
    
    // Inicializo la memoria y los hilos de ejecucion.
    struct mls_instance instance;
    mls_init(seed, instance);

    // Mientras los hilos buscan no tengo nada mas que hacer.
    // Espero a que terminen.

    // Bloqueo la ejecucion hasta que terminen todos los hilos.
    for(int i = 0; i < instance.count_threads; i++)
    {
        if(pthread_join(instance.threads[i], NULL))
        {
            printf("Could not join thread %d\n", i);
            exit(EXIT_FAILURE);
        }
    }

    // Todos los individuos del archivo que estaban como candidatos a ser
    // borrados no se borran. Los marco como disponibles.
    for (int i = 0; i < instance.population_max_size; i++)
    {
        if (instance.population[i].status == SOLUTION__STATUS_TO_DEL)
        {
            instance.population[i].status = SOLUTION__STATUS_READY;
        }
    }
    
    int total_iterations = 0;
    int total_soluciones_no_evolucionadas = 0;
    int total_soluciones_evolucionadas_dominadas = 0;
    int total_re_iterations = 0;

    for (int i = 0; i < instance.count_threads; i++)
    {
        total_iterations += instance.threads_args[i].total_iterations;
        total_soluciones_no_evolucionadas += instance.threads_args[i].total_soluciones_no_evolucionadas;
        total_soluciones_evolucionadas_dominadas += instance.threads_args[i].total_soluciones_evolucionadas_dominadas;
        total_re_iterations += instance.threads_args[i].total_re_iterations;
    }

    fprintf(stdout, "[INFO] Cantidad de iteraciones             : %d\n", total_iterations);
    fprintf(stdout, "[INFO] Cantidad de soluciones en el archivo: %d\n", instance.population_count);
    fprintf(stdout, "[INFO] Cantidad de soluciones no mejoradas : %d\n", total_soluciones_no_evolucionadas);
    fprintf(stdout, "[INFO] Cantidad de soluciones dominadas    : %d\n", total_soluciones_evolucionadas_dominadas);
    fprintf(stdout, "[INFO] Cantidad de re-trabajos             : %d\n", total_re_iterations);

    fprintf(stdout, "== Population =================================================\n");
    for (int i = 0; i < instance.population_max_size; i++)
    {
        // Mostrar la solución: instance.population[i]
    }

    // Libero la memoria pedida en la inicialización.
    mls_finalize(instance);
}


void mls_init(int seed, struct mls_instance &empty_instance)
{
    // Inicializo las estructuras de datos.
    // ...

    // =========================================================================
    // Pido la memoria e inicializo la solucion de partida.
    empty_instance.population_max_size = MAX_ARCHIVE_SIZE;
    empty_instance.population_count = 0;
    empty_instance.global_total_iterations = 0;

    // =========================================================================
    // Inicializo la población.
    empty_instance.population = (struct solution*)malloc(sizeof(struct solution) * empty_instance.population_max_size);
    if (empty_instance.population == NULL)
    {
        fprintf(stderr, "[ERROR] Solicitando memoria para population.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < empty_instance.population_max_size; i++)
    {
        empty_instance.population[i].status = SOLUTION__STATUS_EMPTY;
    }
    
    // =========================================================================
    // Inicializo el archivo.
    empty_instance.archiver_state = (struct aga_state*)malloc(sizeof(struct aga_state));
    archivers_aga_init(&empty_instance);

    // =========================================================================
    // Pedido de memoria para la generacion de numeros aleatorios e inicializo 
    // las semillas.
    srand(seed);
    long int random_seed;

    empty_instance.random_states = (struct cpu_mt_state*)malloc(sizeof(struct cpu_mt_state) * empty_instance.count_threads);

    for (int i = 0; i < empty_instance.count_threads; i++)
    {
        random_seed = rand();
        cpu_mt_init(random_seed, empty_instance.random_states[i]);
    }

    // =========================================================================
    // Creo e inicializo los threads y los mecanismos de sincronizacion.

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
    empty_instance.threads = (pthread_t*)malloc(sizeof(pthread_t) * empty_instance.count_threads);
    empty_instance.threads_args = (struct mls_thread_arg*)malloc(sizeof(struct mls_thread_arg) * empty_instance.count_threads);
    empty_instance.work_type = PALS_CPU_1POP_WORK__INIT;

    for (int i = 0; i < empty_instance.count_threads; i++)
    {
        empty_instance.threads_args[i].thread_idx = i;
        empty_instance.threads_args[i].count_threads = empty_instance.count_threads;

        empty_instance.threads_args[i].max_iterations = MAX_ITERATIONS;
        empty_instance.threads_args[i].global_total_iterations = &(empty_instance.global_total_iterations);
        
        empty_instance.threads_args[i].population = empty_instance.population;
        empty_instance.threads_args[i].population_count = &(empty_instance.population_count);
        empty_instance.threads_args[i].population_max_size = empty_instance.population_max_size;

        empty_instance.threads_args[i].archiver_state = empty_instance.archiver_state;
        empty_instance.threads_args[i].work_type = &(empty_instance.work_type);

        empty_instance.threads_args[i].population_mutex = &(empty_instance.population_mutex);
        empty_instance.threads_args[i].sync_barrier = &(empty_instance.sync_barrier);

        empty_instance.threads_args[i].thread_random_state = &(empty_instance.random_states[i]);

        if (pthread_create(&(empty_instance.threads[i]), NULL, mls_thread,  (void*) &(empty_instance.threads_args[i])))
        {
            printf("Could not create slave thread %d\n", i);
            exit(EXIT_FAILURE);
        }
    }
}


void mls_finalize(struct mls_instance &instance)
{
    free(instance.population);
    free(instance.random_states);
    free(instance.threads);
    free(instance.threads_args);

    archivers_aga_free(&instance);
    
    pthread_mutex_destroy(&(instance.population_mutex));
    pthread_barrier_destroy(&(instance.sync_barrier));
}


void* mls_thread(void *thread_arg)
{
    int rc;

    struct mls_thread_arg *thread_instance;
    thread_instance = (mls_thread_arg*)thread_arg;

    thread_instance->total_iterations = 0;
    thread_instance->total_soluciones_no_evolucionadas = 0;
    thread_instance->total_soluciones_evolucionadas_dominadas = 0;
    thread_instance->total_re_iterations = 0;

    int terminate = 0;
    int work_type = -1;

    int selected_solution_pos = -1;
    int local_iteration_count = 0;

    int candidate_to_del_pos;
    int i;
    int random_sol_index;
    int current_sol_pos;
    int current_sol_index;
    int work_do_iteration;
    int work_iteration_size;
    int search_iteration;
    int mutex_locked;
    int new_solution_eval;

    double random = 0;

    while ((terminate == 0) && (thread_instance->global_total_iterations[0] < thread_instance->max_iterations))
    {
        work_type = *(thread_instance->work_type);

        if (work_type == PALS_CPU_1POP_WORK__EXIT)
        {
            // =================================================================
            // Finalizo la ejecucion del algoritmo!
            terminate = 1;
        }
        else if (work_type == PALS_CPU_1POP_WORK__INIT)
        {
            // =================================================================
            // Inicializo un individuo con una heurística.

            // ...

            // Espero a que los demas hilos terminen.
            rc = pthread_barrier_wait(thread_instance->sync_barrier);
            if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
            {
                printf("Could not wait on barrier\n");
                exit(EXIT_FAILURE);
            }

            // Comienza la busqueda.
            pthread_mutex_lock(thread_instance->population_mutex);
                thread_instance->work_type[0] = PALS_CPU_1POP_WORK__SEARCH;
            pthread_mutex_unlock(thread_instance->population_mutex);

        }
        else if (work_type == PALS_CPU_1POP_WORK__SEARCH)
        {
            // Busqueda...

            // ===============================================================================
            // Busco un lugar libre en la poblacion para clonar un individuo y evolucionarlo
            if (selected_solution_pos == -1) {
                pthread_mutex_lock(thread_instance->population_mutex);
                    candidate_to_del_pos = -1;
                    
                    for (i = 0; (i < thread_instance->population_max_size) && (selected_solution_pos == -1); i++)
                    {
                        if (thread_instance->population[i].status == SOLUTION__STATUS_EMPTY)
                        {
                            // Encontré un espacio libre. Lo reservo.
                            thread_instance->population[i].status = SOLUTION__STATUS_NOT_READY;
                            selected_solution_pos = i;
                        } else if (thread_instance->population[i].status == SOLUTION__STATUS_TO_DEL) {
                            // Encontré una solución marcada para ser borrada.
                            // Sigo buscando por las dudas que haya un lugar libre más adelante.
                            candidate_to_del_pos = i;
                        }
                    }

                    if ((selected_solution_pos == -1) && (candidate_to_del_pos != -1)) {
                        selected_solution_pos = candidate_to_del_pos;
                        thread_instance->population[selected_solution_pos].status = SOLUTION__STATUS_NOT_READY;
                    }
                pthread_mutex_unlock(thread_instance->population_mutex);
            }

            // Y si no encuentro un lugar libre? duermo un rato y vuelvo a probar?
            if (selected_solution_pos == -1)
            {
                // No se que hacer... panico! termino!
                fprintf(stdout, "[ERROR] Hilo finalizado.");
                terminate = 1;
            }
            else
            {
                struct solution *selected_solution;
                selected_solution = &(thread_instance->population[selected_solution_pos]);

                // =================================================================
                // Sorteo la solucion que voy a intentar evolucionar
                random = cpu_mt_generate(*(thread_instance->thread_random_state));
                random_sol_index = (int)floor(random * (*(thread_instance->population_count)));

                pthread_mutex_lock(thread_instance->population_mutex);
                    // Aprovecho y actualizo el contador de iteraciones.
                    thread_instance->global_total_iterations[0] += local_iteration_count;
                    local_iteration_count = 0;

                    // Busco la solución sorteada.
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

                    // Clono la solucion elegida
                    clone_solution(selected_solution, &(thread_instance->population[current_sol_pos]));
                pthread_mutex_unlock(thread_instance->population_mutex);

                // =================================================================
                // Empiezo con la busqueda
                work_do_iteration = 1;
                
                random = cpu_mt_generate(*(thread_instance->thread_random_state));
                work_iteration_size = (int)floor((PALS_CPU_1POP_WORK__THREAD_ITERATIONS / PALS_CPU_1POP_WORK__THREAD_RE_WORK_FACTOR) +
                    (random * (PALS_CPU_1POP_WORK__THREAD_ITERATIONS - (PALS_CPU_1POP_WORK__THREAD_ITERATIONS / PALS_CPU_1POP_WORK__THREAD_RE_WORK_FACTOR))));

                while (work_do_iteration == 1) {
                    // A menos que alguien me indique lo contrario no hay re-trabajo.
                    work_do_iteration = 0;

                    for (search_iteration = 0; search_iteration < work_iteration_size; search_iteration++)
                    {
                        thread_instance->total_iterations++;
                        local_iteration_count++;

                        // =================================================================
                        // Busco y aplico el mejor movimiento en el vecindario de turno
                        // ...
                    } 

                    // Solamente si logro mejorar la solucion, intento agregarla al archivo.
                    if (1 == 1) // TODO!!!
                    {
                        new_solution_eval = 0;

                        // Intento obtener lock de la poblacion.
                        mutex_locked = pthread_mutex_trylock(thread_instance->population_mutex);

                        if (mutex_locked == 0) {
                            // Pude lockear el archivo. Chequeo si la nueva solucion puede ser agregada
                            // al archivo.
                            new_solution_eval = archivers_aga(thread_instance, selected_solution_pos);

                            // Libero el lock.
                            pthread_mutex_unlock(thread_instance->population_mutex);

                            if (new_solution_eval == 1)
                            {
                                // Yupi! La solución quedó agregada al archivo.
                            } else {
                                // Buuuu, la solución no fue agregada.
                                thread_instance->total_soluciones_evolucionadas_dominadas++;
                            }

                            // Libero la posicion seleccionada.
                            selected_solution_pos = -1;
                        } else {
                            // Algun otro thread esta trabajando sobre la población.
                            // Intento hacer otro loop de trabajo y vuelvo a probar.
                            work_do_iteration = 1;
                            
                            random = cpu_mt_generate(*(thread_instance->thread_random_state));
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
    }

    if (selected_solution_pos != -1) {
        if (thread_instance->population[selected_solution_pos].status == SOLUTION__STATUS_NOT_READY) {
            thread_instance->population[selected_solution_pos].status = SOLUTION__STATUS_EMPTY;
        }
    }

    return NULL;
}
