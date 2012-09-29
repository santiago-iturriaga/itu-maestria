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
};

extern struct mls_instance MLS;

/*
 * Ejecuta el algoritmo.
 */
void mls(int seed);

#endif /* MLS__H_ */
