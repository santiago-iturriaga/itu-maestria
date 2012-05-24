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

#ifndef PALS_CPU_2POP_H_
#define PALS_CPU_2POP_H_

#define PALS_CPU_2POP_WORK__TIMEOUT      5
#define PALS_CPU_2POP_WORK__CONVERGENCE  1000
#define PALS_CPU_2POP_WORK__COUNT        1000000

#define PALS_CPU_2POP_WORK__THREAD_CONVERGENCE  50
#define PALS_CPU_2POP_WORK__THREAD_ITERATIONS   250

#define PALS_CPU_2POP_WORK__ELITE_POP_MAX_SIZE 50
#define PALS_CPU_2POP_WORK__POP_SIZE_FACTOR    20

#define PALS_CPU_2POP_WORK__SRC_TASK_NHOOD 12
#define PALS_CPU_2POP_WORK__DST_TASK_NHOOD 6
#define PALS_CPU_2POP_WORK__DST_MACH_NHOOD 2

#define PALS_CPU_2POP_SEARCH_OP__SWAP 0
#define PALS_CPU_2POP_SEARCH_OP__MOVE 1

#define PALS_CPU_2POP_SEARCH__MAKESPAN_GREEDY 0
#define PALS_CPU_2POP_SEARCH__ENERGY_GREEDY   1
#define PALS_CPU_2POP_SEARCH__RANDOM_GREEDY   2

#define PALS_CPU_2POP_WORK__INIT_POP 0
#define PALS_CPU_2POP_WORK__SEARCH   1
#define PALS_CPU_2POP_WORK__WAIT     2
#define PALS_CPU_2POP_WORK__EXIT     3

struct pals_cpu_2pop_instance {
    // Estado del problema.
    struct etc_matrix *etc;
    struct energy_matrix *energy;

    // Referencia a los threads del disponibles.
    pthread_t master_thread;
    pthread_t *slave_threads;
    struct pals_cpu_2pop_thread_arg *slave_threads_args;

	// Espacio de memoria para comunicación con los threads.
	int *slave_work_type;

    struct solution *population;
    int *population_locked;
    int population_count;
    int population_max_size;
	
    struct solution **elite_population;
    int elite_population_count;   

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

	// Statics	
	int total_mater_iteraciones;
	int total_reinicializaciones;
    int total_elite_population_full;
};

struct pals_cpu_2pop_thread_arg {
    // Id del thread actual.
    int thread_idx;
    
    // Estado del problema.
    struct etc_matrix *etc;
    struct energy_matrix *energy;

    // Comunicación con el thread actual.
    struct solution *population;
    int *population_locked;
    int *population_count;
    int population_max_size;
	
	int *work_type;
	
	pthread_mutex_t     *work_type_mutex;
	pthread_mutex_t     *population_mutex;
	sem_t               *new_solutions_sem;
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
    int total_population_full;
    int total_to_delete_solutions;
};

/*
 * Ejecuta el algoritmo.
 */
void pals_cpu_2pop(struct params &input, struct etc_matrix *etc, struct energy_matrix *energy);

/*
 * Reserva e inicializa la memoria con los datos del problema.
 */
void pals_cpu_2pop_init(struct params &input, struct etc_matrix *etc, struct energy_matrix *energy,
    int seed, struct pals_cpu_2pop_instance &empty_instance);

/*
 * Libera la memoria.
 */
void pals_cpu_2pop_finalize(struct pals_cpu_2pop_instance &instance);

/*
 * Ejecuta PALS multi-hilado.
 */
void* pals_cpu_2pop_master_thread(void *thread_arg);
void* pals_cpu_2pop_slave_thread(void *thread_arg);

#endif /* PALS_CPU_H_ */
