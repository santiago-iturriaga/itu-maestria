#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <assert.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <mpi.h>

#include "mls.h"

#include "../config.h"
#include "../random/cpu_mt.h"

#ifdef NONMPI
    #include "../aga/aga.h"
#endif

struct mls_instance MLS;

/*
 * Reserva e inicializa la memoria con los datos del problema.
 */
void mls_init(int seed);

/*
 * Libera la memoria pedida durante la inicialización.
 */
void mls_finalize();

/*
 * Ejecuta un hilo de la búsqueda.
 */
void* mls_thread(void *data);

void mls(int seed, MPI_Comm *mls_comm)
{
    // Inicializo la memoria y los hilos de ejecucion.
    MLS.mls_comm = mls_comm;
    mls_init(seed);

    // Mientras los hilos buscan no tengo nada mas que hacer.
    // Espero a que terminen....

    // Bloqueo la ejecucion hasta que terminen todos los hilos.
    for(int i = 0; i < MLS.count_threads; i++)
    {
        if(pthread_join(MLS.threads[i], NULL))
        {
            fprintf(stderr, "[ERROR][%d] Could not join thread %d\n", world_rank, i);
            exit(EXIT_FAILURE);
        }
    }

    int total_iterations = 0;

    for (int i = 0; i < MLS.count_threads; i++)
    {
        total_iterations += MLS.total_iterations[i];
    }

    fprintf(stdout, "[INFO][%d] Cantidad de iteraciones: %d\n", world_rank, total_iterations);

    // Libero la memoria pedida en la inicialización.
    mls_finalize();
}


void mls_init(int seed)
{
    // =========================================================================
    // Inicializo las semillas para la generación de numeros aleatorios.
    srand(seed);
    long int random_seed;

    for (int i = 0; i < MLS.count_threads; i++)
    {
        random_seed = rand();
        cpu_mt_init(random_seed, MLS.random_states[i]);
    }

    // =========================================================================
    // Creo e inicializo los threads y los mecanismos de sincronizacion.

    for (int i = 0; i < MLS.count_threads; i++) {
        if (pthread_mutex_init(&(MLS.work_type_mutex[i]), NULL))
        {
            fprintf(stderr, "[ERROR][%d] Could not create work type mutex\n", world_rank);
            exit(EXIT_FAILURE);
        }
    }

    if (pthread_mutex_init(&(MLS.mpi_mutex), NULL))
    {
        fprintf(stderr, "[ERROR][%d] Could not create mpi mutex\n", world_rank);
        exit(EXIT_FAILURE);
    }
        
    if (pthread_barrier_init(&(MLS.sync_barrier), NULL, MLS.count_threads))
    {
        fprintf(stderr, "[ERROR][%d] Could not create sync barrier\n", world_rank);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < MLS.count_threads; i++)
    {
        MLS.threads_id[i] = i;
        MLS.work_type[i] = MLS__INIT;

        if (pthread_create(&MLS.threads[i], NULL, mls_thread,  (void*) &MLS.threads_id[i]))
        {
            fprintf(stderr, "[ERROR][%d] Could not create slave thread %d\n", world_rank, i);
            exit(EXIT_FAILURE);
        }
    }
}

void mls_finalize()
{
    for (int i = 0; i < MLS.count_threads; i++) {
        pthread_mutex_destroy(&(MLS.work_type_mutex[i]));
    }
    
    pthread_mutex_destroy(&(MLS.mpi_mutex));
    pthread_barrier_destroy(&(MLS.sync_barrier));
}

void* mls_thread(void *data)
{
    int thread_id = *((int*)data);

    #ifndef NDEBUG
        fprintf(stderr, "[DEBUG][%d] Thread id: %d\n", world_rank, thread_id);
    #endif

    MLS.total_iterations[thread_id] = 0;

    int terminate = 0;
    int work_iteration_size;
    int search_iteration;
    double random = 0;
    int work_type;

    while ((terminate == 0) && (MLS.total_iterations[thread_id] < MLS.max_iterations))
    {
        pthread_mutex_lock(&MLS.work_type_mutex[thread_id]);
            work_type = MLS.work_type[thread_id];
        pthread_mutex_unlock(&MLS.work_type_mutex[thread_id]);

        if (work_type == MLS__EXIT)
        {
            // =================================================================
            // Finalizo la ejecucion del hilo!
            terminate = 1;
        }
        else if (work_type == MLS__INIT)
        {
            #ifndef NDEBUG
                fprintf(stderr, "[DEBUG] MLS__INIT\n");
            #endif

            // Inicializo el NS3 para este thread.
            #ifndef LOCAL
                MLS.simul[thread_id] = ns3AEDBRestrictedCall();
            #endif

            // =================================================================
            // Inicializo un individuo con una heurística.
            MLS.population[thread_id].borders_threshold = cpu_mt_generate(MLS.random_states[thread_id]) * (MLS.ubound_border_threshold - MLS.lbound_border_threshold) + MLS.lbound_border_threshold;
            MLS.population[thread_id].margin_forwarding = cpu_mt_generate(MLS.random_states[thread_id]) * (MLS.ubound_margin_threshold - MLS.lbound_margin_threshold) + MLS.lbound_margin_threshold;
            MLS.population[thread_id].min_delay = cpu_mt_generate(MLS.random_states[thread_id]) * (MLS.ubound_min_delay - MLS.lbound_min_delay) + MLS.lbound_min_delay;
            MLS.population[thread_id].max_delay = cpu_mt_generate(MLS.random_states[thread_id]) * (MLS.ubound_max_delay - MLS.lbound_max_delay) + MLS.lbound_max_delay;
            MLS.population[thread_id].neighbors_threshold = cpu_mt_generate(MLS.random_states[thread_id]) * (MLS.ubound_neighbors_threshold - MLS.lbound_neighbors_threshold) + MLS.lbound_neighbors_threshold;

            #ifndef LOCAL
                // Call the ns3 function, and the results for the three objectives and the time (which is used as a constraint) are put in aux
                double *aux;
                aux = MLS.simul[thread_id].RunExperimentAEDBRestricted(MLS.number_devices, MLS.simul_runs,
                    MLS.population[thread_id].min_delay, MLS.population[thread_id].max_delay,
                    MLS.population[thread_id].borders_threshold, MLS.population[thread_id].margin_forwarding,
                    MLS.population[thread_id].neighbors_threshold);

                MLS.population[thread_id].energy = aux[0];
                MLS.population[thread_id].coverage = aux[1];
                MLS.population[thread_id].nforwardings = aux[2];
                MLS.population[thread_id].time = aux[3];
                
                free(aux);
            #else
                MLS.population[thread_id].energy = cpu_mt_generate(MLS.random_states[thread_id]);
                MLS.population[thread_id].coverage = cpu_mt_generate(MLS.random_states[thread_id]);
                MLS.population[thread_id].nforwardings = cpu_mt_generate(MLS.random_states[thread_id]);
                MLS.population[thread_id].time = cpu_mt_generate(MLS.random_states[thread_id]);
            #endif

            #ifndef NDEBUG
                fprintf(stderr, "[DEBUG][%d] Thread %d starting solution:\n", world_rank, thread_id);
                show_solution(&MLS.population[thread_id]);
            #endif

            // Envío la solución computada por la heurística a AGA.
            pthread_mutex_lock(&MLS.mpi_mutex);
                #ifndef NONMPI
                    #ifdef MPI_MODE_STANDARD
                        MPI_Send(&MLS.population[thread_id], 1, mpi_solution_type, AGA__PROCESS_RANK, AGA__NEW_SOL_MSG, MPI_COMM_WORLD);
                    #endif

                    #ifdef MPI_MODE_SYNC
                        MPI_Ssend(&MLS.population[thread_id], 1, mpi_solution_type, AGA__PROCESS_RANK, AGA__NEW_SOL_MSG, MPI_COMM_WORLD);
                    #endif

                    #ifdef MPI_MODE_BUFFERED
                        MPI_Bsend(&MLS.population[thread_id], 1, mpi_solution_type, AGA__PROCESS_RANK, AGA__NEW_SOL_MSG, MPI_COMM_WORLD);
                    #endif
                #else
                    // NONMPI
                    MLS.population[thread_id].status = SOLUTION__STATUS_NEW;
                    for (int i = 0; (i < AGA__MAX_ARCHIVE_SIZE) && (MLS.population[thread_id].status == SOLUTION__STATUS_NEW); i++) {
                        if (AGA.population[i].status == SOLUTION__STATUS_EMPTY) {
                            // Encontré una posición libre. Agrego la solución al archivo acá.
                            clone_solution(&AGA.population[i], &MLS.population[thread_id]);
                            rc = archivers_aga_add(i);
                            MLS.population[thread_id].status = SOLUTION__STATUS_READY;
                        }
                    }
                #endif
            pthread_mutex_unlock(&MLS.mpi_mutex);

            MLS.total_iterations[thread_id] = 0;

            // Comienza la busqueda.
            pthread_mutex_lock(&MLS.work_type_mutex[thread_id]);
                MLS.work_type[thread_id] = MLS__SEARCH;
            pthread_mutex_unlock(&MLS.work_type_mutex[thread_id]);
        }
        else if (work_type == MLS__SEARCH) {
            double delta;
            double alfa = 0.2;
            int rand_op;

            // =================================================================
            // Empiezo con la busqueda
            random = cpu_mt_generate(MLS.random_states[thread_id]);
            work_iteration_size = (int)(MLS__THREAD_FIXED_ITERS + (random * MLS__THREAD_RANDOM_ITERS));

            for (search_iteration = 0; search_iteration < work_iteration_size; search_iteration++) {
                MLS.total_iterations[thread_id]++;

                // =================================================================
                // RUSO

                //rand_op = cpu_mt_generate_int(MLS.random_states[thread_id],NUM_LS_OPERATORS-1);
                rand_op = cpu_mt_generate(MLS.random_states[thread_id]) * NUM_LS_OPERATORS;

                switch(rand_op){
                    case LS_ENERGY :
                    case LS_FORWARDING :
                        // Reduce borders_threshold
                        //delta = MLS.population[thread_id+MLS.count_threads].borders_threshold - MLS.population[thread_id].borders_threshold;
                        delta = MLS.population[thread_id].borders_threshold * 0.1;
                        
                        if (delta > 0){
                            MLS.population[thread_id].borders_threshold -= alfa * delta * cpu_mt_generate(MLS.random_states[thread_id]);
                        } else {
                            MLS.population[thread_id].borders_threshold += alfa * delta * cpu_mt_generate(MLS.random_states[thread_id]);
                        }
                        // Reduce neighbors_threshold
                        delta = MLS.population[thread_id+MLS.count_threads].neighbors_threshold - MLS.population[thread_id].neighbors_threshold;
                        if (delta > 0){
                            MLS.population[thread_id].neighbors_threshold -= floor(alfa * delta * cpu_mt_generate(MLS.random_states[thread_id]));
                        } else {
                            MLS.population[thread_id].neighbors_threshold += floor(alfa * delta * cpu_mt_generate(MLS.random_states[thread_id]));
                        }
                        
                        if (MLS.population[thread_id].borders_threshold < MLS.lbound_border_threshold) 
                            MLS.population[thread_id].borders_threshold = MLS.lbound_border_threshold;
                        
                        break;
                    case LS_COVERAGE :
                        // Augment neighbors_threshold
                        //delta = MLS.population[thread_id+MLS.count_threads].neighbors_threshold - MLS.population[thread_id].neighbors_threshold;
                        delta = MLS.population[thread_id].neighbors_threshold * 0.1;
                        
                        if (delta > 0){
                            MLS.population[thread_id].neighbors_threshold += floor(alfa * delta * cpu_mt_generate(MLS.random_states[thread_id]));
                        } else {
                            MLS.population[thread_id].neighbors_threshold -= floor(alfa * delta * cpu_mt_generate(MLS.random_states[thread_id]));
                        }
    
                        if (MLS.population[thread_id].neighbors_threshold < MLS.lbound_neighbors_threshold) 
                            MLS.population[thread_id].neighbors_threshold = MLS.lbound_neighbors_threshold;
                        
                        break;
                    case LS_TIME :
                        delta = MLS.population[thread_id].max_delay - MLS.population[thread_id].min_delay;
                        
                        if (cpu_mt_generate(MLS.random_states[thread_id]) < 0.5){
                            // Reduce max delay
                            MLS.population[thread_id].max_delay -= alfa * delta * cpu_mt_generate(MLS.random_states[thread_id]);
                        } else {
                            // Augment min delay
                            MLS.population[thread_id].min_delay += alfa * delta * cpu_mt_generate(MLS.random_states[thread_id]);
                        }
                        
                        if (MLS.population[thread_id].min_delay < MLS.lbound_min_delay) MLS.population[thread_id].min_delay = MLS.lbound_min_delay;
                        if (MLS.population[thread_id].max_delay < MLS.lbound_max_delay) MLS.population[thread_id].max_delay = MLS.lbound_max_delay;
                        
                        break;
                }
            }

            #ifndef LOCAL
                // Call the ns3 function, and the results for the three objectives and the time (which is used as a constraint) are put in aux
                double *aux;
                aux = MLS.simul[thread_id].RunExperimentAEDBRestricted(MLS.number_devices, MLS.simul_runs,
                    MLS.population[thread_id].min_delay, MLS.population[thread_id].max_delay,
                    MLS.population[thread_id].borders_threshold, MLS.population[thread_id].margin_forwarding,
                    MLS.population[thread_id].neighbors_threshold);

                MLS.population[thread_id].energy = aux[0];
                MLS.population[thread_id].coverage = aux[1];
                MLS.population[thread_id].nforwardings = aux[2];
                MLS.population[thread_id].time = aux[3];
                
                free(aux);
            #else
                MLS.population[thread_id].energy = cpu_mt_generate(MLS.random_states[thread_id]);
                MLS.population[thread_id].coverage = cpu_mt_generate(MLS.random_states[thread_id]);
                MLS.population[thread_id].nforwardings = cpu_mt_generate(MLS.random_states[thread_id]);
                MLS.population[thread_id].time = cpu_mt_generate(MLS.random_states[thread_id]);
            #endif
            
            #ifndef NDEBUG
                fprintf(stderr, "[DEBUG][%d] Thread %d found:\n", world_rank, thread_id);
                show_solution(&MLS.population[thread_id]);
            #endif

            pthread_mutex_lock(&MLS.mpi_mutex);
                #ifndef NONMPI
                    #ifdef MPI_MODE_STANDARD
                        MPI_Send(&MLS.population[thread_id], 1, mpi_solution_type, AGA__PROCESS_RANK, AGA__NEW_SOL_MSG, MPI_COMM_WORLD);
                    #endif

                    #ifdef MPI_MODE_SYNC
                        MPI_Ssend(&MLS.population[thread_id], 1, mpi_solution_type, AGA__PROCESS_RANK, AGA__NEW_SOL_MSG, MPI_COMM_WORLD);
                    #endif

                    #ifdef MPI_MODE_BUFFERED
                        MPI_Bsend(&MLS.population[thread_id], 1, mpi_solution_type, AGA__PROCESS_RANK, AGA__NEW_SOL_MSG, MPI_COMM_WORLD);
                    #endif
                #else
                    // NONMPI
                    MLS.population[thread_id].status = SOLUTION__STATUS_NEW;
                    for (int i = 0; (i < AGA__MAX_ARCHIVE_SIZE) && (MLS.population[thread_id].status == SOLUTION__STATUS_NEW); i++) {
                        if (AGA.population[i].status == SOLUTION__STATUS_EMPTY) {
                            // Encontré una posición libre. Agrego la solución al archivo acá.
                            clone_solution(&AGA.population[i], &MLS.population[thread_id]);
                            rc = archivers_aga_add(i);
                            MLS.population[thread_id].status = SOLUTION__STATUS_READY;
                        }
                    }
                #endif
            pthread_mutex_unlock(&MLS.mpi_mutex);
        }
    }

    return NULL;
}
