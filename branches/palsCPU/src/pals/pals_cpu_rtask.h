/*
 * pals.h
 *
 *  Created on: Jul 27, 2011
 *      Author: santiago
 */

#include <pthread.h>
#include <semaphore.h>

#include "../etc_matrix.h"
#include "../energy_matrix.h"
#include "../solution.h"

#ifndef PALS_CPU_RTASK_H_
#define PALS_CPU_RTASK_H_

#define PALS_CPU_RTASK_MASTER_RANDOM_NUMBERS 3
#define PALS_CPU_RTASK_SLAVE_RANDOM_NUMBERS  5

#define PALS_CPU_RTASK_WORK__TIMEOUT      5
#define PALS_CPU_RTASK_WORK__CONVERGENCE  100
#define PALS_CPU_RTASK_WORK__RESET_POP    50

#define PALS_CPU_RTASK_WORK__THREAD_CONVERGENCE  3
#define PALS_CPU_RTASK_WORK__THREAD_ITERATIONS   9

#define PALS_CPU_RTASK_SEARCH_OP__SWAP 0
#define PALS_CPU_RTASK_SEARCH_OP__MOVE 1

#define PALS_CPU_RTASK_SEARCH__MAKESPAN_GREEDY 0
#define PALS_CPU_RTASK_SEARCH__ENERGY_GREEDY   1
#define PALS_CPU_RTASK_SEARCH__RANDOM_GREEDY   2

#define PALS_CPU_RTASK_WORK__INIT_POP 0
#define PALS_CPU_RTASK_WORK__SEARCH   1
#define PALS_CPU_RTASK_WORK__WAIT     2
#define PALS_CPU_RTASK_WORK__EXIT     3

#define PALS_CPU_RTASK_WORK__ELITE_POP_MAX_SIZE 10
#define PALS_CPU_RTASK_WORK__POP_MAX_SIZE  25

#define PALS_CPU_RTASK_WORK__SRC_TASK_NHOOD 10
#define PALS_CPU_RTASK_WORK__DST_TASK_NHOOD 10
#define PALS_CPU_RTASK_WORK__DST_MACH_NHOOD 10

struct pals_cpu_rtask_instance {
    // Estado del problema.
    struct etc_matrix *etc;
    struct energy_matrix *energy;

    // Referencia a los threads del disponibles.
    pthread_t master_thread;
    pthread_t *slave_threads;
    struct pals_cpu_rtask_thread_arg *slave_threads_args;

	// Espacio de memoria para comunicación con los threads.
	int *slave_work_type;

    struct solution *population;
    int *population_locked;
    int population_count;  
	
    struct solution **elite_population;
    int elite_population_count;   
    
	pthread_mutex_t population_mutex;
	sem_t new_solutions_sem;
	pthread_barrier_t sync_barrier;

	// Estado de los generadores aleatorios.
    struct cpu_rand_state *random_states;
	
	// Parámetros de ejecución.
	int count_threads;	

	// Statics	
	int total_reinicializaciones;
    int total_elite_population_full;
};

struct pals_cpu_rtask_thread_arg {
    // Id del thread actual.
    int thread_idx;
    
    // Estado del problema.
    struct etc_matrix *etc;
    struct energy_matrix *energy;

    // Comunicación con el thread actual.
    struct solution *population;
    int *population_locked;
    int *population_count;
	
	int *work_type;
	
	pthread_mutex_t *population_mutex;
	sem_t *new_solutions_sem;
	pthread_barrier_t *sync_barrier;
	
	// Estado del generador aleatorio para el thread actual.
    struct cpu_rand_state *thread_random_state;
	
	// Statics
	int total_iterations;
    int total_makespan_greedy_searches;
    int total_energy_greedy_searches;
    int total_random_greedy_searches;
	int total_swaps;
    int total_moves;
    int total_population_full;
};

/*
 * Ejecuta el algoritmo.
 */
void pals_cpu_rtask(struct params &input, struct etc_matrix *etc, struct energy_matrix *energy);

/*
 * Reserva e inicializa la memoria con los datos del problema.
 */
void pals_cpu_rtask_init(struct params &input, struct etc_matrix *etc, struct energy_matrix *energy,
    int seed, struct pals_cpu_rtask_instance &empty_instance);

/*
 * Libera la memoria.
 */
void pals_cpu_rtask_finalize(struct pals_cpu_rtask_instance &instance);

/*
 * Ejecuta PALS multi-hilado.
 */
void* pals_cpu_rtask_master_thread(void *thread_arg);
void* pals_cpu_rtask_slave_thread(void *thread_arg);

#endif /* PALS_CPU_H_ */
