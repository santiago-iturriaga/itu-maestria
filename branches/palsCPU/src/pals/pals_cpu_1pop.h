/*
 * pals.h
 *
 *  Created on: Jul 27, 2011
 *      Author: santiago
 */

#include <pthread.h>
#include <semaphore.h>

#include "../config.h"
#include "../etc_matrix.h"
#include "../energy_matrix.h"
#include "../solution.h"

#ifndef PALS_CPU_1POP_H_
#define PALS_CPU_1POP_H_

#define PALS_CPU_1POP_WORK__TIMEOUT             20
#define PALS_CPU_1POP_WORK__THREAD_ITERATIONS   200
#define PALS_CPU_1POP_WORK__POP_SIZE_FACTOR     4

#define PALS_CPU_1POP_WORK__SRC_TASK_NHOOD      8
#define PALS_CPU_1POP_WORK__DST_TASK_NHOOD      8
#define PALS_CPU_1POP_WORK__DST_MACH_NHOOD      8

#define PALS_CPU_1POP_SEARCH_OP__SWAP           0
#define PALS_CPU_1POP_SEARCH_OP__MOVE           1

#define PALS_CPU_1POP_SEARCH_OP_BALANCE__SWAP   0.80
#define PALS_CPU_1POP_SEARCH_OP_BALANCE__MOVE   0.20

#define PALS_CPU_1POP_SEARCH__MAKESPAN_GREEDY   0
#define PALS_CPU_1POP_SEARCH__ENERGY_GREEDY     1
#define PALS_CPU_1POP_SEARCH__RANDOM_GREEDY     2

#define PALS_CPU_1POP_SEARCH__MAKESPAN_GREEDY_PSEL_BEST     0.15
#define PALS_CPU_1POP_SEARCH__MAKESPAN_GREEDY_PSEL_WORST    0.15
#define PALS_CPU_1POP_SEARCH__ENERGY_GREEDY_PSEL_BEST       0.15
#define PALS_CPU_1POP_SEARCH__ENERGY_GREEDY_PSEL_WORST      0.15

#define PALS_CPU_1POP_SEARCH_BALANCE__MAKESPAN  0.40
#define PALS_CPU_1POP_SEARCH_BALANCE__ENERGY    0.40

#define PALS_CPU_1POP_WORK__INIT                0
#define PALS_CPU_1POP_WORK__SEARCH              1
#define PALS_CPU_1POP_WORK__EXIT                3

struct pals_cpu_1pop_instance {
    // Estado del problema.
    struct etc_matrix *etc;
    struct energy_matrix *energy;

    // Referencia a los threads del disponibles.
    pthread_t *threads;
    struct pals_cpu_1pop_thread_arg *threads_args;

    struct solution *population;
    int population_count;
    int population_max_size;

    int work_type;

    pthread_mutex_t     work_type_mutex;
    pthread_mutex_t     population_mutex;
    sem_t               new_solutions_sem;
    pthread_barrier_t   sync_barrier;

    // Estado de los generadores aleatorios.
    #ifdef CPU_MERSENNE_TWISTER
    struct cpu_mt_state *random_states;
    #else
    struct cpu_rand_state *random_states;
    #endif

    // Parámetros de ejecución.
    int count_threads;
};

struct pals_cpu_1pop_thread_arg {
    // Id del thread actual.
    int thread_idx;

    // Estado del problema.
    struct etc_matrix *etc;
    struct energy_matrix *energy;

    // Comunicación con el thread actual.
    struct solution *population;
    int *population_count;
    int population_max_size;

    int count_threads;
    int *work_type;

    pthread_mutex_t     *population_mutex;
    pthread_barrier_t   *sync_barrier;

    // Estado del generador aleatorio para el thread actual.
    #ifdef CPU_MERSENNE_TWISTER
    struct cpu_mt_state *thread_random_state;
    #else
    struct cpu_rand_state *thread_random_state;
    #endif

    // Statics
    int total_iterations;

    int total_makespan_greedy_searches;
    int total_energy_greedy_searches;
    int total_random_greedy_searches;
    int total_swaps;
    int total_moves;

    int total_success_makespan_greedy_searches;
    int total_success_energy_greedy_searches;
    int total_success_random_greedy_searches;

    int total_population_full;

    timespec ts_start;
    timespec ts_last_found;
};

/*
 * Ejecuta el algoritmo.
 */
void pals_cpu_1pop(struct params &input, struct etc_matrix *etc, struct energy_matrix *energy);

/*
 * Reserva e inicializa la memoria con los datos del problema.
 */
void pals_cpu_1pop_init(struct params &input, struct etc_matrix *etc, struct energy_matrix *energy,
    int seed, struct pals_cpu_1pop_instance &empty_instance);

/*
 * Libera la memoria.
 */
void pals_cpu_1pop_finalize(struct pals_cpu_1pop_instance &instance);

/*
 * Ejecuta PALS multi-hilado.
 */
void* pals_cpu_1pop_thread(void *thread_arg);

#endif /* PALS_CPU_H_ */
