/*
 * pals.h
 *
 *  Created on: Jul 27, 2011
 *      Author: santiago
 */

#include <pthread.h>

#include "../etc_matrix.h"
#include "../solution.h"

#define PALS_CPU_RTASK_SWAP 0
#define PALS_CPU_RTASK_MOVE 1

#ifndef PALS_CPU_RTASK_H_
#define PALS_CPU_RTASK_H_

#define PALS_CPU_RTASK_WORK__DO_GREEDY_SEARCH 0
#define PALS_CPU_RTASK_WORK__DO_RANDOM_SEARCH 1
#define PALS_CPU_RTASK_WORK__DO_EXIT 2

#define PALS_CPU_RTASK_WORK__GREEDY_CONV 5
#define PALS_CPU_RTASK_WORK__RANDOM_CONV 5
#define PALS_CPU_RTASK_WORK__RANDOM_MAX_ITERS 25


struct pals_cpu_rtask_instance {
    // Estado del problema.
    struct matrix *etc_matrix;
    struct solution *current_solution;
    struct solution *best_solution;

    // Referencia a los threads del disponibles.
    pthread_t *master_thread;
    pthread_t *slave_threads;
    struct pals_cpu_rtask_thread_arg *slave_threads_args;

	// Espacio de memoria para comunicación con los threads.
	int work_type;
	pthread_barrier_t *sync_barrier;

	// Estado de los generadores aleatorios.
    struct cpu_rand_state *random_states;
    double *random_numbers;
	
	// Espacio de memoria para almacenar los mejores movimientos encontrados en cada iteración.
	int *move_type;
	int *origin;
	int *destination;
	float *delta;
	
	// Parámetros de ejecución.
	int count_threads;
	int count_loops;
	int count_evals;
	
	// Cantidad de movimientos probados por iteración.
    long total_evals;
	
	// Cantidad de resultados obtenidos por iteración.
	// (¿siempre es igual a cantidad de bloques del kernel?)
	int result_count;
	
	// Inner state...
	char *__result_task_history;
	char *__result_machine_history;
};

struct pals_cpu_rtask_thread_arg {
    // Id del thread actual.
    int thread_idx;

	// Parámetros de ejecución.
	int count_loops;
	int count_evals;
    
    // Estado del problema.
    struct matrix *etc_matrix;
    struct solution *current_solution;

    // Comunicación con el thread actual.
	int *work_type;
	pthread_barrier_t *sync_barrier;
	
	// Estado del generador aleatorio para el thread actual.
    struct cpu_rand_state *thread_random_state;
    double *thread_random_numbers;

    // Mejor movimiento del thread actual.
	int *thread_move_type;
	int *thread_origin;
	int *thread_destination;
	float *thread_delta;
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
