/*
 * pals.h
 *
 *  Created on: Jul 27, 2011
 *      Author: santiago
 */

#include <pthread.h>

#include "../etc_matrix.h"
#include "../solution.h"

#ifndef PALS_CPU_RTASK_H_
#define PALS_CPU_RTASK_H_

#define PALS_CPU_RTASK_SWAP 0
#define PALS_CPU_RTASK_MOVE 1

#define PALS_CPU_RTASK_MASTER_RANDOMS 3

#define PALS_CPU_RTASK_WORK__DO_MAKESPAN_GREEDY_SEARCH 0
#define PALS_CPU_RTASK_WORK__DO_ENERGY_GREEDY_SEARCH   1
#define PALS_CPU_RTASK_WORK__DO_RANDOM_GREEDY_SEARCH   2
#define PALS_CPU_RTASK_WORK__DO_EXIT                   99

#define PALS_CPU_RTASK_WORK__CONVERGENCE 100

#define PALS_CPU_RTASK_WORK__ELITE_POP_MAX_SIZE 10
#define PALS_CPU_RTASK_WORK__ELITE_POP_EMPTY    0
#define PALS_CPU_RTASK_WORK__ELITE_POP_SOL      1

#define PALS_CPU_RTASK_WORK__SRC_TASK_NHOOD 10
#define PALS_CPU_RTASK_WORK__DST_TASK_NHOOD 10

struct pals_cpu_rtask_instance {
    // Estado del problema.
    struct matrix *etc_matrix;
    struct solution *initial_solution;

    // Referencia a los threads del disponibles.
    pthread_t *master_thread;
    pthread_t *slave_threads;
    struct pals_cpu_rtask_thread_arg *slave_threads_args;

	// Espacio de memoria para comunicación con los threads.
	int *slave_work_type;
	int *slave_work_solution_idx;
	
    struct solution *elite_population;
    int *elite_population_status;
    int elite_population_count;   
    
	pthread_barrier_t *sync_barrier;

	// Estado de los generadores aleatorios.
    struct cpu_rand_state *random_states;
	double master_random_numbers[PALS_CPU_RTASK_MASTER_RANDOMS];
    double *slave_random_numbers;
	
	// Espacio de memoria para almacenar los mejores movimientos encontrados en cada iteración.
	int *move_type;
	int *origin;
	int *destination;
	float *delta_makespan;
	float *delta_ct;
	float *delta_energy;
	
	// Parámetros de ejecución.
	int count_threads;
	
	// Cantidad de resultados obtenidos por iteración.
	int result_count;
	
	// Inner state...
	char *__result_task_history;
	char *__result_machine_history;
	
	// Statics
    int total_iterations;
    int last_elite_found_on_iter;
    int total_makespan_greedy_searches;
    int total_energy_greedy_searches;
    int total_random_greedy_searches;
	long total_swaps;
    long total_moves;
};

struct pals_cpu_rtask_thread_arg {
    // Id del thread actual.
    int thread_idx;
    
    // Estado del problema.
    struct matrix *etc_matrix;

    // Comunicación con el thread actual.
    struct solution *population;
    int *population_status;
    int *population_count;   
    
	int *work_solution_idx;
	int *work_type;
	
	pthread_barrier_t *sync_barrier;
	
	// Estado del generador aleatorio para el thread actual.
    struct cpu_rand_state *thread_random_state;
    double *thread_random_numbers;

    // Mejor movimiento del thread actual.
	int *thread_move_type;
	int *thread_origin;
	int *thread_destination;
	float *thread_delta_makespan;
	float *thread_delta_ct;
	float *thread_delta_energy;
};

/*
 * Ejecuta el algoritmo.
 * Búsqueda masivamente paralela sobre un subdominio del problema. 
 * Se sortea el subdominio por tarea.
 */
void pals_cpu_rtask(struct params &input, struct matrix *etc_matrix, struct solution *current_solution);

/*
 * Reserva e inicializa la memoria con los datos del problema.
 */
void pals_cpu_rtask_init(struct params &input, struct matrix *etc_matrix, struct solution *s, int seed, struct pals_cpu_rtask_instance &empty_instance);

/*
 * Libera la memoria.
 */
void pals_cpu_rtask_finalize(struct pals_cpu_rtask_instance &instance);

/*
 * Ejecuta PALS.
 */
void* pals_cpu_rtask_master_thread(void *thread_arg);
void* pals_cpu_rtask_slave_thread(void *thread_arg);

#endif /* PALS_CPU_H_ */
