#include <pthread.h>

#include "../config.h"
#include "../solution.h"
#include "../random/cpu_mt.h"

#ifndef MLS__H_
#define MLS__H_

// Configuracion
#define MLS__THREAD_FIXED_ITERS     250
#define MLS__THREAD_RANDOM_ITERS    250

// Constantes
#define MLS__INIT       0
#define MLS__SEARCH     1
#define MLS__EXIT       3

struct mls_instance {
    // Referencia a los threads del disponibles.
    pthread_t threads[MLS__MAX_THREADS];
    int threads_id[MLS__MAX_THREADS];
    int work_type[MLS__MAX_THREADS];
    struct solution population[MLS__MAX_THREADS];

    // Config. NS3
    int number_devices;
    int simul_runs;

    // Rango de búsqueda
    double lbound_min_delay;
    double ubound_min_delay;
    double lbound_max_delay;
    double ubound_max_delay;
    double lbound_border_threshold;
    double ubound_border_threshold;
    double lbound_margin_threshold;
    double ubound_margin_threshold;
    int lbound_neighbors_threshold;
    int ubound_neighbors_threshold;
        
    // Sync
    pthread_mutex_t work_type_mutex[MLS__MAX_THREADS];
    pthread_barrier_t sync_barrier;

    // Estado de los generadores aleatorios.
    struct cpu_mt_state random_states[MLS__MAX_THREADS];

    // Condicion de parada
    int max_iterations;

    // Parámetros de ejecución.
    int count_threads;

    // Statics
    int total_iterations[MLS__MAX_THREADS];
    
    // MPI
    MPI_Comm *mls_comm;
};

extern struct mls_instance MLS;

/*
 * Ejecuta el algoritmo.
 */
void mls(int seed, MPI_Comm *mls_comm);

#endif /* MLS__H_ */
