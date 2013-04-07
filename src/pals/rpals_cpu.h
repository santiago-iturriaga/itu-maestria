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
#include "../random/cpu_rand.h"
#include "../random/cpu_drand48.h"
#include "../random/cpu_mt.h"

#ifndef RPALS_CPU_H_
#define RPALS_CPU_H_

#define RPALS_CPU_WORK__THREAD_ITERATIONS       650

#define RPALS_CPU_WORK__SRC_TASK_NHOOD      27
#define RPALS_CPU_WORK__DST_TASK_NHOOD      15
#define RPALS_CPU_WORK__DST_MACH_NHOOD      15

#define RPALS_CPU_SEARCH_OP__SWAP           0
#define RPALS_CPU_SEARCH_OP__MOVE           1

#define RPALS_CPU_SEARCH_OP_BALANCE__SWAP   0.80
#define RPALS_CPU_SEARCH_OP_BALANCE__MOVE   0.20

#define RPALS_CPU_SEARCH__MAKESPAN_GREEDY   0
#define RPALS_CPU_SEARCH__ENERGY_GREEDY     1
#define RPALS_CPU_SEARCH__RANDOM_GREEDY     2

#define RPALS_CPU_SEARCH__MAKESPAN_GREEDY_PSEL_BEST     0.90
#define RPALS_CPU_SEARCH__MAKESPAN_GREEDY_PSEL_WORST    0.80
#define RPALS_CPU_SEARCH__ENERGY_GREEDY_PSEL_BEST       0.15
#define RPALS_CPU_SEARCH__ENERGY_GREEDY_PSEL_WORST      0.15

#define RPALS_CPU_SEARCH_BALANCE__MAKESPAN  0.55
#define RPALS_CPU_SEARCH_BALANCE__ENERGY    0.45

struct rpals_cpu_instance {
    // Estado del problema.
    struct etc_matrix *etc;
    struct energy_matrix *energy;

    struct solution population[2];
    int best_solution;

    int total_iterations;

    // Estado de los generadores aleatorios.
    #ifdef CPU_MERSENNE_TWISTER
        struct cpu_mt_state random_state;
    #endif
    #ifdef CPU_RAND
        struct cpu_rand_state random_state;
    #endif
    #ifdef CPU_DRAND48
        struct cpu_drand48_state random_state;
    #endif
    
    int max_time_secs;
    int max_iterations;
};

/*
 * Ejecuta el algoritmo.
 */
void rpals_cpu(struct params &input, struct etc_matrix *etc, struct energy_matrix *energy);

/*
 * Reserva e inicializa la memoria con los datos del problema.
 */
void rpals_cpu_init(struct params &input, struct etc_matrix *etc, struct energy_matrix *energy,
    int seed, struct rpals_cpu_instance &empty_instance);

/*
 * Libera la memoria.
 */
void rpals_cpu_finalize(struct rpals_cpu_instance &instance);

#endif /* PALS_CPU_H_ */
