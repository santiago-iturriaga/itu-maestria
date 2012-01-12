#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <assert.h>
#include <string.h>
#include <pthread.h>

#include "../config.h"
#include "../utils.h"

#include "../random/cpu_rand.h"

#include "pals_cpu_rtask.h"

#define        PALS_RTASK_RANDS 6144*20
#define PALS_GPU_RTASK__THREADS 128

void* pals_cpu_rtask_thread(void *thread_arg)
{	
/*
	const short mov_type = (short)(block_idx & 0x1);

	const unsigned int random1 = gpu_random_numbers[block_idx];
	const unsigned int random2 = gpu_random_numbers[block_idx + 1];

	__shared__ short block_operations[PALS_GPU_RTASK__THREADS];
	__shared__ ushort block_threads[PALS_GPU_RTASK__THREADS];
	__shared__ ushort block_loops[PALS_GPU_RTASK__THREADS];
	__shared__ float block_deltas[PALS_GPU_RTASK__THREADS];
	
	for (ushort loop = 0; loop < loops_count; loop++) {
		// Tipo de movimiento.
		if (mov_type == 0) { // Comparación a nivel de bit para saber si es par o impar.
			// Si es impar... 
			// Movimiento SWAP.
		
			ushort task_x, task_y;
			ushort machine_a, machine_b;
		
			float machine_a_ct_old, machine_b_ct_old;
			float machine_a_ct_new, machine_b_ct_new;
		
			float delta;
			delta = 0.0;
		
			// ================= Obtengo las tareas sorteadas.
			task_x = (random1 + loop) % tasks_count;
				
			task_y = ((random2 >> 1) + (loop * block_dim)  + thread_idx) % (tasks_count - 1);	
			if (task_y >= task_x) task_y++;
		
			// ================= Obtengo las máquinas a las que estan asignadas las tareas.
			machine_a = gpu_task_assignment[task_x]; // Máquina a.	
			machine_b = gpu_task_assignment[task_y]; // Máquina b.	

			if (machine_a != machine_b) {
				// Calculo el delta del swap sorteado.
			
				// Máquina 1.
				machine_a_ct_old = gpu_machine_compute_time[machine_a];
					
				machine_a_ct_new = machine_a_ct_old;
				machine_a_ct_new = machine_a_ct_new - gpu_etc_matrix[(machine_a * tasks_count) + task_x]; // Resto del ETC de x en a.
				machine_a_ct_new = machine_a_ct_new + gpu_etc_matrix[(machine_a * tasks_count) + task_y]; // Sumo el ETC de y en a.
			
				// Máquina 2.
				machine_b_ct_old = gpu_machine_compute_time[machine_b];

				machine_b_ct_new = machine_b_ct_old;
				machine_b_ct_new = machine_b_ct_new - gpu_etc_matrix[(machine_b * tasks_count) + task_y]; // Resto el ETC de y en b.
				machine_b_ct_new = machine_b_ct_new + gpu_etc_matrix[(machine_b * tasks_count) + task_x]; // Sumo el ETC de x en b.

				if ((machine_a_ct_new > current_makespan) || (machine_b_ct_new > current_makespan)) {
					// Luego del movimiento aumenta el makespan. Intento desestimularlo lo más posible.
					if (machine_a_ct_new > current_makespan) delta = delta + (machine_a_ct_new - current_makespan);
					if (machine_b_ct_new > current_makespan) delta = delta + (machine_b_ct_new - current_makespan);
				} else if ((machine_a_ct_old+1 >= current_makespan) || (machine_b_ct_old+1 >= current_makespan)) {	
					// Antes del movimiento una las de máquinas definía el makespan. Estos son los mejores movimientos.
				
					if (machine_a_ct_old+1 >= current_makespan) {
						delta = delta + (machine_a_ct_new - machine_a_ct_old);
					} else {
						delta = delta + 1/(machine_a_ct_new - machine_a_ct_old);
					}
				
					if (machine_b_ct_old+1 >= current_makespan) {
						delta = delta + (machine_b_ct_new - machine_b_ct_old);
					} else {
						delta = delta + 1/(machine_b_ct_new - machine_b_ct_old);
					}
				} else {
					// Ninguna de las máquinas intervenía en el makespan. Intento favorecer lo otros movimientos.
					delta = delta + (machine_a_ct_new - machine_a_ct_old);
					delta = delta + (machine_b_ct_new - machine_b_ct_old);
					delta = 1 / delta;
				}
			}

			if ((loop == 0) || (block_deltas[thread_idx] > delta)) {
				block_operations[thread_idx] = PALS_GPU_RTASK_SWAP;
				block_threads[thread_idx] = (short)thread_idx;
				block_loops[thread_idx] = loop;
				block_deltas[thread_idx] = delta;
			}
		} else {
			// Si es par...
			// Movimiento MOVE.
		
			ushort task_x;
			ushort machine_a, machine_b;
		
			float machine_a_ct_old, machine_b_ct_old;
			float machine_a_ct_new, machine_b_ct_new;

			float delta;
			delta = 0.0;
		
			// ================= Obtengo la tarea sorteada, la máquina a la que esta asignada,
			// ================= y el compute time de la máquina.
			task_x = (random1 + loop) % tasks_count;
			machine_a = gpu_task_assignment[task_x]; // Máquina a.
			machine_a_ct_old = gpu_machine_compute_time[machine_a];	
							
			// ================= Obtengo la máquina destino sorteada.
			machine_b = ((random2 >> 1) + (loop * block_dim) + thread_idx) % (machines_count - 1);
			if (machine_b >= machine_a) machine_b++;
		
			machine_b_ct_old = gpu_machine_compute_time[machine_b];
		
			// Calculo el delta del swap sorteado.
			machine_a_ct_new = machine_a_ct_old - gpu_etc_matrix[(machine_a * tasks_count) + task_x]; // Resto del ETC de x en a.		
			machine_b_ct_new = machine_b_ct_old + gpu_etc_matrix[(machine_b * tasks_count) + task_x]; // Sumo el ETC de x en b.

			if (machine_b_ct_new > current_makespan) {
				// Luego del movimiento aumenta el makespan. Intento desestimularlo lo más posible.
				delta = delta + (machine_b_ct_new - current_makespan);
			} else if (machine_a_ct_old+1 >= current_makespan) {	
				// Antes del movimiento una las de máquinas definía el makespan. Estos son los mejores movimientos.
				delta = delta + (machine_a_ct_new - machine_a_ct_old);
				delta = delta + 1/(machine_b_ct_new - machine_b_ct_old);
			} else {
				// Ninguna de las máquinas intervenía en el makespan. Intento favorecer lo otros movimientos.
				delta = delta + (machine_a_ct_new - machine_a_ct_old);
				delta = delta + (machine_b_ct_new - machine_b_ct_old);
				delta = 1 / delta;
			}
			
			if ((loop == 0) || (block_deltas[thread_idx] > delta)) {
				block_operations[thread_idx] = PALS_GPU_RTASK_MOVE;
				block_threads[thread_idx] = (short)thread_idx;
				block_loops[thread_idx] = loop;
				block_deltas[thread_idx] = delta;
			}
		}
	}
	
	__syncthreads();

	// Aplico reduce para quedarme con el mejor delta.
	int pos;
	for (int i = 1; i < block_dim; i *= 2) {
		pos = 2 * i * thread_idx;
	
		if (pos < block_dim) {
			if (block_deltas[pos] > block_deltas[pos + i]) {			
				block_operations[pos] = block_operations[pos + i];
				block_loops[pos] = block_loops[pos + i];
				block_threads[pos] = block_threads[pos + i];
				block_deltas[pos] = block_deltas[pos + i];
			}
		}
	
		__syncthreads();
	}
	
	if (thread_idx == 0) {
		gpu_best_movements[block_idx * 3] = (int)block_operations[0]; // Best movement operation.
		gpu_best_movements[(block_idx * 3) + 1] = (int)block_threads[0]; // Best movement thread index.
		gpu_best_movements[(block_idx * 3) + 2] = (int)block_loops[0]; // Best movement loop index.
		gpu_best_deltas[block_idx] = block_deltas[0];  // Best movement delta.
	}
	*/
}

void pals_cpu_rtask_init(struct matrix *etc_matrix, struct solution *s, int seed,
	struct pals_cpu_rtask_instance &empty_instance) {
	
	// Asignación del paralelismo del algoritmo.
	empty_instance.count_threads = 4;
	empty_instance.count_loops = 4;
	empty_instance.count_evals = 128;
	
	// Cantidad total de movimientos a evaluar.
	empty_instance.total_evals = empty_instance.count_threads * empty_instance.count_loops * empty_instance.count_evals;
	
	// Cantidad de resultados retornados por iteración.
	empty_instance.result_count = empty_instance.count_threads;
	
	if (DEBUG) {
		fprintf(stdout, "[INFO] Seed                           : %d\n", seed);
		fprintf(stdout, "[INFO] Number of threads              : %d\n", empty_instance.count_threads);
		fprintf(stdout, "[INFO] Loops per thread               : %d\n", empty_instance.count_loops);
		fprintf(stdout, "[INFO] Evaluations per loop           : %d\n", empty_instance.count_evals);
		fprintf(stdout, "[INFO] Total evaluations              : %ld\n", empty_instance.total_evals);
	}

	// =========================================================================
	// Pedido de memoria para la generación de numeros aleatorios.
	
	timespec ts_1;
	timming_start(ts_1);
	
	srand(seed);
	long int random_seed;
	
	empty_instance.random_states = (struct cpu_rand_state*)malloc(sizeof(struct cpu_rand_state) * empty_instance.count_threads);
	
	for (int i = 0; i < empty_instance.count_threads; i++) {
	    random_seed = rand();
	    cpu_rand_init(random_seed, empty_instance.random_states[i]);
	}
	
	timming_end(".. cpu_rand_buffers", ts_1);
	
	// =========================================================================
	// Pedido de memoria para almacenar los mejores movimientos de cada iteración.
	
	empty_instance.move_type = (int*)malloc(sizeof(int) * empty_instance.result_count);
	empty_instance.origin = (int*)malloc(sizeof(int) * empty_instance.result_count);
	empty_instance.destination = (int*)malloc(sizeof(int) * empty_instance.result_count);
	empty_instance.delta = (float*)malloc(sizeof(int) * empty_instance.result_count);

	// =========================================================================
	// Creo los threads del sistema.
	
	timespec ts_threads;
	timming_start(ts_threads);
	
	empty_instance.threads = (pthread_t*)malloc(sizeof(pthread_t) * empty_instance.count_threads);
	empty_instance.threads_args = (struct pals_cpu_rtask_thread_arg*)malloc(
	    sizeof(struct pals_cpu_rtask_thread_arg) * empty_instance.count_threads);
	
	for (int i = 0; i < empty_instance.count_threads; i++) {
   		empty_instance.threads_args[i].thread_idx = i;
   		
        empty_instance.threads_args[i].etc_matrix = etc_matrix;
        empty_instance.threads_args[i].current_solution = s;
        
    	empty_instance.threads_args[i].work_type = &(empty_instance.work_type);
        empty_instance.threads_args[i].thread_random_state = &(empty_instance.random_states[i]);
	    empty_instance.threads_args[i].thread_move_type = &(empty_instance.move_type[i]);
	    empty_instance.threads_args[i].thread_origin = &(empty_instance.origin[i]);
	    empty_instance.threads_args[i].thread_destination = &(empty_instance.destination[i]);
	    empty_instance.threads_args[i].thread_delta = &(empty_instance.delta[i]);
   		
	    pthread_create(&(empty_instance.threads[i]), NULL, pals_cpu_rtask_thread, 
            (void*) &(empty_instance.threads_args[i]));
	}

	timming_end(".. thread creation", ts_threads);
}

void pals_cpu_rtask_finalize(struct pals_cpu_rtask_instance &instance) {
    free(instance.random_states);
	free(instance.move_type);
	free(instance.origin);
	free(instance.destination);
	free(instance.delta);
	free(instance.threads);	
	free(instance.threads_args);
}

void pals_cpu_rtask_search(struct pals_cpu_rtask_instance &instance) {
	// Timming -----------------------------------------------------
	timespec ts_pals_pre;
	timming_start(ts_pals_pre);
	// Timming -----------------------------------------------------
	
	// Timming -----------------------------------------------------
	timming_end(".. pals_cpu_rtask_pals_pre", ts_pals_pre);
	// Timming -----------------------------------------------------
	
	// ==============================================================================
	// Ejecución del algoritmo.
	// ==============================================================================	
	
	// Timming -----------------------------------------------------
	timespec ts_pals;
	timming_start(ts_pals);
	// Timming -----------------------------------------------------

    /*	
	pals_cpu_rtask_thread(
		instance.loops,
		etc_matrix->machines_count,
		etc_matrix->tasks_count,
		s->makespan,
		instance.gpu_etc_matrix, 
		instance.gpu_task_assignment, 
		instance.gpu_machine_compute_time, 
		gpu_random_numbers,
		instance.gpu_best_movements, 
		instance.gpu_best_deltas);

	cudaError_t e;
    e = cudaGetLastError();
    if (e != cudaSuccess) {
		fprintf(stderr, "[ERROR] Failure in kernel call.\n%s\n", cudaGetErrorString(e));
		exit(EXIT_FAILURE);
    }

	// Pido el espacio de memoria para obtener los resultados desde la gpu.
	int *best_movements = (int*)malloc(sizeof(int) * instance.blocks * 3);
	float *best_deltas = (float*)malloc(sizeof(float) * instance.blocks);
	int *rands_nums = (int*)malloc(sizeof(int) * instance.blocks * 2);

	// Copio los mejores movimientos desde el dispositivo.
	if (cudaMemcpyAsync(best_movements, instance.gpu_best_movements, sizeof(int) * instance.blocks * 3,
		cudaMemcpyDeviceToHost, 0) != cudaSuccess) {
		
		fprintf(stderr, "[ERROR] Copiando los mejores movimientos al host (best_swaps).\n");
		exit(EXIT_FAILURE);
	}
	
	if (cudaMemcpyAsync(best_deltas, instance.gpu_best_deltas, sizeof(float) * instance.blocks, 
		cudaMemcpyDeviceToHost, 0) != cudaSuccess) {
		
		fprintf(stderr, "[ERROR] Copiando los mejores movimientos al host (best_swaps_delta).\n");
		exit(EXIT_FAILURE);
	}

	if (cudaMemcpyAsync(rands_nums, gpu_random_numbers, sizeof(int) * instance.blocks * 2, 
		cudaMemcpyDeviceToHost, 0) != cudaSuccess) {
		
		fprintf(stderr, "[ERROR] Copiando al host los números aleatorios sorteados.\n");
		exit(EXIT_FAILURE);
	}

	cudaThreadSynchronize();*/

	// Timming -----------------------------------------------------
	timming_end(".. pals_cpu_rtask_pals", ts_pals);
	// Timming -----------------------------------------------------

	// =====================================================================
	// Se cargan los resultados a la respuesta.
	// (lo mejor sería usar la GPU para generar el resultado).
	// =====================================================================

	// Timming -----------------------------------------------------
	timespec ts_pals_post;
	timming_start(ts_pals_post);
	// Timming -----------------------------------------------------
	
	// Busco el block que encontró el mejor movimiento.
	int best_block_idx = 0;
	
	/*
	for (int i = 1; i < instance.blocks; i++) {
		if (best_deltas[i] < best_deltas[best_block_idx]) {
			best_block_idx = i;
		}

		
		//if (DEBUG) { 
		//	fprintf(stdout, ".. id=%d, eval=%f.\n", i, best_deltas[i]);
		//}
	}
	
	for (int i = 0; i < instance.blocks; i++) {
		// Calculo cuales fueron los elementos modificados en ese mejor movimiento.	
		int block_idx = (i + best_block_idx) % instance.blocks;

		int movement_idx = block_idx * 3;
		int move_type = best_movements[movement_idx];
		int thread_idx = best_movements[movement_idx + 1];
		int loop_idx = best_movements[movement_idx + 2];

		float delta = best_deltas[block_idx];

		int random_idx = block_idx;
		int random1 = rands_nums[random_idx];
		int random2 = rands_nums[random_idx + 1];

		if (move_type == PALS_GPU_RTASK_SWAP) { // Movement type: SWAP
			ushort task_x = (ushort)((random1 + loop_idx) % etc_matrix->tasks_count);

		    ushort task_y = (ushort)(((random2 >> 1) + (loop_idx * instance.threads) + thread_idx) % (etc_matrix->tasks_count - 1));
		    if (task_y >= task_x) task_y++;

			result.move_type[i] = (short)move_type; // SWAP
			result.origin[i] = task_x;
			result.destination[i] = task_y;
			result.delta[i] = delta;
		
			// =======> DEBUG
			if (DEBUG) { 
				ushort machine_a = s->task_assignment[task_x];
				ushort machine_b = s->task_assignment[task_y];

				fprintf(stdout, "[DEBUG] Task %d in %d swaps with task %d in %d. Delta %f.\n",
					task_x, machine_a, task_y, machine_b, delta);
			}
			// <======= DEBUG
		} else if (move_type == PALS_GPU_RTASK_MOVE) { // Movement type: MOVE
			ushort task_x = (ushort)((random1 + loop_idx) % etc_matrix->tasks_count);
			ushort machine_a = s->task_assignment[task_x];

		    ushort machine_b = (ushort)(((random2 >> 1) + (loop_idx * instance.threads) + thread_idx) % (etc_matrix->machines_count - 1));
		    if (machine_b >= machine_a) machine_b++;

			result.move_type[i] = (short)move_type; // MOVE
			result.origin[i] = task_x;
			result.destination[i] = machine_b;
			result.delta[i] = delta;
		
			// =======> DEBUG
			if (DEBUG) {
				fprintf(stdout, "[DEBUG] Task %d in %d is moved to machine %d. Delta %f.\n",
					task_x, machine_a, machine_b, delta);
			}
			// <======= DEBUG
		}
	}*/
	
	// Timming -----------------------------------------------------
	timming_end(".. pals_cpu_rtask_pals_post", ts_pals_post);
	// Timming -----------------------------------------------------
}

void pals_cpu_rtask(struct params &input, struct matrix *etc_matrix, struct solution *current_solution) {	
	// ==============================================================================
	// PALS aleatorio por tarea.
	// ==============================================================================
	
	// Timming -----------------------------------------------------
	timespec ts_init;
	timming_start(ts_init);
	// Timming -----------------------------------------------------

	struct pals_cpu_rtask_instance instance;
			
	// Inicializo la memoria en el dispositivo.
	pals_cpu_rtask_init(etc_matrix, current_solution, input.seed, instance);

	// Timming -----------------------------------------------------
	timming_end(">> pals_cpu_rtask_init", ts_init);
	// Timming -----------------------------------------------------

	// ===========> DEBUG
	if (DEBUG) {
		validate_solution(etc_matrix, current_solution);
	}
	// <=========== DEBUG
	
	float makespan_inicial = current_solution->makespan;
	
	char result_task_history[etc_matrix->tasks_count];
	char result_machine_history[etc_matrix->machines_count];

	int increase_depth;
	increase_depth = 0;
	
	long cantidad_swaps = 0;
	long cantidad_movs = 0;

	int convergence_flag;
	convergence_flag = 0;
	
	struct solution *best_solution = create_empty_solution(etc_matrix);
	clone_solution(etc_matrix, best_solution, current_solution);
	
	int best_solution_iter = -1;
	
	int iter;
	for (iter = 0; (iter < PALS_COUNT) && (convergence_flag == 0); iter++) {
		if (DEBUG) fprintf(stdout, "[INFO] Iteracion %d =====================\n", iter);

		// Timming -----------------------------------------------------
		timespec ts_search;
		timming_start(ts_search);
		// Timming -----------------------------------------------------

		pals_cpu_rtask_search(instance);

		// Timming -----------------------------------------------------
		timming_end(">> pals_cpu_rtask_search", ts_search);
		// Timming -----------------------------------------------------

		// Timming -----------------------------------------------------
		timespec ts_post;
		timming_start(ts_post);
		// Timming -----------------------------------------------------

		// Aplico el mejor movimiento.
		memset(result_task_history, 0, etc_matrix->tasks_count);
		memset(result_machine_history, 0, etc_matrix->machines_count);
		
		long cantidad_swaps_iter, cantidad_movs_iter;
		cantidad_swaps_iter = 0;
		cantidad_movs_iter = 0;
			
		for (int result_idx = 0; result_idx < instance.result_count; result_idx++) {
			//if (DEBUG) fprintf(stdout, "[DEBUG] Movement %d, delta = %f.\n", result_idx, result.delta[result_idx]);
		
			if (instance.delta[result_idx] < 0.0) { //|| (increase_depth < 50)) {
				if (instance.move_type[result_idx] == PALS_CPU_RTASK_SWAP) {
					int task_x = instance.origin[result_idx];
					int task_y = instance.destination[result_idx];
			
					int machine_a = current_solution->task_assignment[instance.origin[result_idx]];
					int machine_b = current_solution->task_assignment[instance.destination[result_idx]];
			
					/*if (DEBUG) fprintf(stdout, "        (swap) Task %d in %d swaps with task %d in %d. Delta %f.\n",
						result.origin[result_idx], machine_a, result.destination[result_idx], machine_b, result.delta[result_idx]);*/
			
					if ((result_task_history[task_x] == 0) && (result_task_history[task_y] == 0) &&
						(result_machine_history[machine_a] == 0) && (result_machine_history[machine_b] == 0))	{
			
						cantidad_swaps_iter++;
			
						result_task_history[task_x] = 1;
						result_task_history[task_y] = 1;
						result_machine_history[machine_a] = 1;
						result_machine_history[machine_b] = 1;
						
						/*if (DEBUG) {
							fprintf(stdout, ">> [pre-update]:\n");
							fprintf(stdout, "   machine_a: %d, old_machine_a_ct: %f.\n", machine_a, current_solution->machine_compute_time[machine_a]);
							fprintf(stdout, "   machine_b: %d, old_machine_b_ct: %f.\n", machine_b, current_solution->machine_compute_time[machine_b]);
						}*/
			
						// Actualizo la asignación de cada tarea en el host.
						current_solution->task_assignment[task_x] = machine_b;
						current_solution->task_assignment[task_y] = machine_a;
			
						// Actualizo los compute time de cada máquina luego del move en el host.
						current_solution->machine_compute_time[machine_a] = 
							current_solution->machine_compute_time[machine_a] +
							get_etc_value(etc_matrix, machine_a, task_y) - 
							get_etc_value(etc_matrix, machine_a, task_x);

						current_solution->machine_compute_time[machine_b] = 
							current_solution->machine_compute_time[machine_b] +
							get_etc_value(etc_matrix, machine_b, task_x) - 
							get_etc_value(etc_matrix, machine_b, task_y);

						/*if (DEBUG) {
							fprintf(stdout, ">> [update]:\n");
							fprintf(stdout, "   task_x: %d, task_x_machine: %d.\n", task_x, machine_b);
							fprintf(stdout, "   task_y: %d, task_y_machine: %d.\n", task_y, machine_a);
							fprintf(stdout, "   machine_a: %d, machine_a_ct: %f.\n", machine_a, current_solution->machine_compute_time[machine_a]);
							fprintf(stdout, "   machine_b: %d, machine_b_ct: %f.\n", machine_b, current_solution->machine_compute_time[machine_b]);
							fprintf(stdout, "   old_makespan: %f.\n", current_solution->makespan);
						}*/
					} else {
						//if (DEBUG) fprintf(stdout, "[DEBUG] Lo ignoro porque una tarea o máquina de este movimiento ya fue modificada.\n");
					}
				} else if (instance.move_type[result_idx] == PALS_CPU_RTASK_MOVE) {
					int task_x = instance.origin[result_idx];
					int machine_a = current_solution->task_assignment[task_x];
					int machine_b = instance.destination[result_idx];

					/*if (DEBUG) fprintf(stdout, "        (move) Task %d in %d is moved to machine %d. Delta %f.\n",
						result.origin[result_idx], machine_a, result.destination[result_idx], result.delta[result_idx]);*/
					
					if ((result_task_history[task_x] == 0) &&
	 					(result_machine_history[machine_a] == 0) &&
						(result_machine_history[machine_b] == 0))	{
			
						cantidad_movs_iter++;
			
						result_task_history[task_x] = 1;
						result_machine_history[machine_a] = 1;
						result_machine_history[machine_b] = 1;
					
						/*if (DEBUG) {
							fprintf(stdout, ">> [pre-update]:\n");
							fprintf(stdout, "   machine_a: %d, old_machine_a_ct: %f.\n", machine_a, current_solution->machine_compute_time[machine_a]);
							fprintf(stdout, "   machine_b: %d, old_machine_b_ct: %f.\n", machine_b, current_solution->machine_compute_time[machine_b]);
						}*/
				
						current_solution->task_assignment[task_x] = machine_b;
					
						// Actualizo los compute time de cada máquina luego del move en el host.
						current_solution->machine_compute_time[machine_a] = 
							current_solution->machine_compute_time[machine_a] - 
							get_etc_value(etc_matrix, machine_a, task_x);

						current_solution->machine_compute_time[machine_b] = 
							current_solution->machine_compute_time[machine_b] +
							get_etc_value(etc_matrix, machine_b, task_x);
				
						/*if (DEBUG) {
							fprintf(stdout, ">> [update]:\n");
							fprintf(stdout, "   task_x: %d, task_x_machine: %d.\n", task_x, machine_b);
							fprintf(stdout, "   machine_a: %d, machine_a_ct: %f.\n", machine_a, current_solution->machine_compute_time[machine_a]);
							fprintf(stdout, "   machine_b: %d, machine_b_ct: %f.\n", machine_b, current_solution->machine_compute_time[machine_b]);
							fprintf(stdout, "   old_makespan: %f.\n", current_solution->makespan);
						}*/
					} else {
						//if (DEBUG) fprintf(stdout, "[DEBUG] Lo ignoro porque una tarea o máquina de este movimiento ya fue modificada.\n");
					}
				}
			}
		}

		if ((cantidad_movs_iter > 0) || (cantidad_swaps_iter > 0)) {
			// Actualiza el makespan de la solución.
			// Si cambio el makespan, busco el nuevo makespan.
			int machine = 0;		
			current_solution->makespan = current_solution->machine_compute_time[0];

			for (int i = 1; i < etc_matrix->machines_count; i++) {
				if (current_solution->makespan < current_solution->machine_compute_time[i]) {
					current_solution->makespan = current_solution->machine_compute_time[i];
					machine = i;
				}
			}
			
			if (current_solution->makespan < best_solution->makespan) {
				clone_solution(etc_matrix, best_solution, current_solution);
				best_solution_iter = iter;
			}

			if (DEBUG) {
				fprintf(stdout, "   swaps performed  : %ld.\n", cantidad_swaps_iter);
				fprintf(stdout, "   movs performed   : %ld.\n", cantidad_movs_iter);
			}
			
			cantidad_swaps += cantidad_swaps_iter;
			cantidad_movs += cantidad_movs_iter;
		}
		
		if (best_solution_iter == iter) {
			increase_depth = 0;
		
			if (DEBUG) {
				fprintf(stdout, "   makespan improved: %f.\n", current_solution->makespan);
			}
		} else {
			increase_depth++;

			if (DEBUG) {
				fprintf(stdout, "   makespan unchanged: %f (%d).\n", current_solution->makespan, increase_depth);
			}
		}

		if (increase_depth >= 500) {
			/*if (DEBUG) fprintf(stdout, "[DEBUG] Increase depth on iteration %d.\n", iter);
		
			instance.blocks += 8;

			if ((instance.blocks == 96) && (instance.loops == 32)) {
				instance.blocks = 32;
				instance.loops = 64;
			}

			fprintf(stdout, "[DEBUG] REINIT! Blocks = %d, Loops = %d.\n", instance.blocks, instance.loops);

			if ((instance.blocks == 96) && (instance.loops = 64)) {
                                convergence_flag = 1;
                                if (DEBUG) fprintf(stdout, "[DEBUG] Convergence detected! Iteration: %d.\n", iter);
			} else {
				pals_cpu_rtask_reinit(instance, result);
			}*/

			convergence_flag = 1;
			increase_depth = 0;
		}

		// Timming -----------------------------------------------------
		timming_end(">> pals_cpu_rtask_post", ts_post);
		// Timming -----------------------------------------------------
	}
	
	// Timming -----------------------------------------------------
	timespec ts_finalize;
	timming_start(ts_finalize);
	// Timming -----------------------------------------------------

	clone_solution(etc_matrix, current_solution, best_solution);

	if (DEBUG) {	
		fprintf(stdout, "[DEBUG] Total iterations       : %d.\n", iter);
		fprintf(stdout, "[DEBUG] Iter. best sol. found  : %d.\n", best_solution_iter);

		fprintf(stdout, "[DEBUG] Total swaps performed  : %ld.\n", cantidad_swaps);
		fprintf(stdout, "[DEBUG] Total movs performed   : %ld.\n", cantidad_movs);

		fprintf(stdout, "[DEBUG] Current threads count  : %d.\n", instance.count_threads);
		fprintf(stdout, "[DEBUG] Current loops count    : %d.\n", instance.count_loops);
	}
	
	// Reconstruye el compute time de cada máquina.
	// NOTA: tengo que hacer esto cada tanto por errores acumulados en el redondeo.
	for (int i = 0; i < etc_matrix->machines_count; i++) {
		current_solution->machine_compute_time[i] = 0.0;
	}
	
	for (int i = 0; i < etc_matrix->tasks_count; i++) {
		int assigned_machine = current_solution->task_assignment[i];
	
		current_solution->machine_compute_time[assigned_machine] =
			current_solution->machine_compute_time[assigned_machine] + 
			get_etc_value(etc_matrix, assigned_machine, i);
	}	
	
	// Actualiza el makespan de la solución.
	current_solution->makespan = current_solution->machine_compute_time[0];
	for (int i = 1; i < etc_matrix->machines_count; i++) {
		if (current_solution->makespan < current_solution->machine_compute_time[i]) {
			current_solution->makespan = current_solution->machine_compute_time[i];
		}
	}
	
	// ===========> DEBUG
	if (DEBUG) {
		validate_solution(etc_matrix, current_solution);
	}
	// <=========== DEBUG
	
	if (DEBUG) {
		fprintf(stdout, "[DEBUG] Viejo makespan: %f\n", makespan_inicial);
		fprintf(stdout, "[DEBUG] Nuevo makespan: %f\n", current_solution->makespan);
	} else {
		if (!OUTPUT_SOLUTION) fprintf(stdout, "%f\n", current_solution->makespan);
		fprintf(stderr, "CANT_ITERACIONES|%d\n", iter);
		fprintf(stderr, "BEST_FOUND|%d\n", best_solution_iter);
                fprintf(stderr, "TOTAL_SWAPS|%ld\n", cantidad_swaps);
                fprintf(stderr, "TOTAL_MOVES|%ld\n", cantidad_movs);
	}

	// Libero la memoria del dispositivo.
	pals_cpu_rtask_finalize(instance);
	
	// Timming -----------------------------------------------------
	timming_end(">> pals_cpu_rtask_finalize", ts_finalize);
	// Timming -----------------------------------------------------		
}

