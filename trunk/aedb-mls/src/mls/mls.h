#include <pthread.h>
#include <semaphore.h>

#include "../config.h"
#include "../solution.h"

#ifndef MLS__H_
#define MLS__H_

// Configuracion
#define MLS__THREAD_ITERATIONS       650
#define MLS__THREAD_RE_WORK_FACTOR   14

// Constantes
#define MLS__INIT                0
#define MLS__SEARCH              1
#define MLS__EXIT                3

struct mls_instance {
    // Referencia a los threads del disponibles.
    pthread_t *threads;
    struct mls_thread_arg *threads_args;

    struct solution *population;
    int population_count;
    int population_max_size;

    struct aga_state *archiver_state;

    int work_type;
    int global_total_iterations;

    pthread_mutex_t     population_mutex;
    pthread_barrier_t   sync_barrier;

    // Estado de los generadores aleatorios.
    struct cpu_mt_state *random_states;

    // Parámetros de ejecución.
    int count_threads;
};

struct mls_thread_arg {
    // Id del thread actual.
    int thread_idx;

    int count_threads;
    int *work_type;
    int *global_total_iterations;
    int max_iterations;

    // Comunicación con el thread actual.
    struct solution *population;
    int *population_count;
    int population_max_size;

    struct aga_state *archiver_state;

    pthread_mutex_t     *population_mutex;
    pthread_barrier_t   *sync_barrier;

    // Estado del generador aleatorio para el thread actual.
    struct cpu_mt_state *thread_random_state;

    // Statics
    int total_iterations;
    int total_soluciones_no_evolucionadas;
    int total_soluciones_evolucionadas_dominadas;
    int total_re_iterations;
};

/*
 * Ejecuta el algoritmo.
 */
void mls();

/*
 * Reserva e inicializa la memoria con los datos del problema.
 */
void mls_init(int seed, struct mls_instance &empty_instance);

/*
 * Libera la memoria pedida durante la inicialización.
 */
void mls_finalize(struct mls_instance &instance);

/*
 * Ejecuta un hilo de la búsqueda.
 */
void* mls_thread(void *thread_arg);

#endif /* MLS__H_ */
