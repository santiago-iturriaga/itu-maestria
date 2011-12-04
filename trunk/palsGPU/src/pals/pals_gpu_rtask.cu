#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <limits.h>
#include <assert.h>

#include "../config.h"
#include "../utils.h"
#include "../random/RNG_rand48.h"

#include "pals_gpu_rtask.h"

#define PALS_GPU_RTASK__BLOCKS 			128
#define PALS_GPU_RTASK__THREADS 		128

__global__ void pals_rtask_kernel(
	int machines_count, int tasks_count, float current_makespan,
	float *gpu_etc_matrix, int *gpu_task_assignment, float *gpu_machine_compute_time, 
	int *gpu_random_numbers, ushort *gpu_best_swaps, float *gpu_best_swaps_delta)
{
	unsigned int thread_idx = threadIdx.x;
	unsigned int block_idx = blockIdx.x;
	
	//unsigned int mov_type = block_idx & 0x1;
	unsigned int mov_type = 0;

	__shared__ ushort block_swaps[PALS_GPU_RTASK__THREADS];
	__shared__ float block_swaps_delta[PALS_GPU_RTASK__THREADS];

	__shared__ int random1, random2;
	
	if (threadIdx.x == 0) {
		random1 = gpu_random_numbers[block_idx];
		random2 = gpu_random_numbers[block_idx + 1];
	}
	
	__syncthreads();
							
	// Tipo de movimiento.	
	if (mov_type == 0) { // Comparación a nivel de bit para saber si es par o impar.
		// Si es impar... 
		// Movimiento SWAP.
		
		int task_x, task_y;
		int machine_a, machine_b;
		
		float machine_a_ct_old, machine_b_ct_old;
		float machine_a_ct_new, machine_b_ct_new;
		
		float eval;
		eval = 0.0;
		
		// ================= Obtengo las tareas sorteadas.
		// TODO: OPTIMIZAR MODULOS!!!
		task_x = random1 % tasks_count;
				
		task_y = (random2 % (tasks_count - 1 - PALS_GPU_RTASK__THREADS)) + thread_idx;	
		
		if (task_y >= task_x) task_y++;
		task_y = task_y % tasks_count;
		
		// ================= Obtengo las máquinas a las que estan asignadas las tareas.
		machine_a = gpu_task_assignment[task_x]; // Máquina a.	
		machine_b = gpu_task_assignment[task_y]; // Máquina b.	

		if (machine_a != machine_b) {
			// Calculo el delta del swap sorteado.
			
			// Máquina 1.
			machine_a_ct_old = gpu_machine_compute_time[task_x];
					
			machine_a_ct_new = machine_a_ct_old - gpu_etc_matrix[(machine_a * tasks_count) + task_x]; // Resto del ETC de x en a.
			machine_a_ct_new = machine_a_ct_new + gpu_etc_matrix[(machine_a * tasks_count) + task_y]; // Sumo el ETC de y en a.
			
			// Máquina 2.
			machine_b_ct_old = gpu_machine_compute_time[task_y];

			machine_b_ct_new = machine_b_ct_old - gpu_etc_matrix[(machine_b * tasks_count) + task_y]; // Resto el ETC de y en b.
			machine_b_ct_new = machine_b_ct_new + gpu_etc_matrix[(machine_b * tasks_count) + task_x]; // Sumo el ETC de x en b.

			if ((machine_a_ct_new > current_makespan) || (machine_b_ct_new > current_makespan)) {
				if (machine_a_ct_new > current_makespan) eval = eval + (machine_a_ct_new - current_makespan);
				if (machine_b_ct_new > current_makespan) eval = eval + (machine_b_ct_new - current_makespan);
			} else {
				eval = eval + (machine_a_ct_new - current_makespan);
				eval = eval + (machine_b_ct_new - current_makespan);
			}
		}

		block_swaps[thread_idx] = (ushort)((PALS_GPU_RTASK_SWAP * PALS_GPU_RTASK__THREADS) + thread_idx);
		block_swaps_delta[thread_idx] = eval;
	} else {
		// Si es par...
		// Movimiento MOVE.
		
		// ================= Obtengo la tarea sorteada, la máquina a la que esta asignada,
		// ================= y el compute time de la máquina.
		/*raux1 = raux1 % tasks_count;
		int_aux1 = gpu_task_assignment[raux1]; // Máquina a.
		float_aux1 = gpu_machine_compute_time[int_aux1];	
							
		// ================= Obtengo la máquina destino sorteada.
		raux2 = raux2 % (machines_count - 1 - PALS_GPU_RTASK__THREADS);
		raux2 = raux2 + thread_idx;	
		if (raux2 >= int_aux1) raux2 = raux2 + 1;

		int_aux2 = raux2 % machines_count;
		float_aux2 = gpu_machine_compute_time[int_aux2];
		
		// Calculo el delta del swap sorteado.
		delta = float_aux1 - gpu_etc_matrix[(int_aux1 * tasks_count) + raux1]; // Resto del ETC de x en a.
		
		// Obtengo la diferencia en la máquina A.
		if (delta > current_makespan) {
			float_aux1 = delta - current_makespan;
		} else if ((float_aux1 + 1 >= current_makespan) && (delta < current_makespan)) { // sumo 1 por problemas de redondeo... funciona? 
			float_aux1 = delta - current_makespan;
		} else {
			float_aux1 = 0.0;
		}
			
		delta = float_aux2 + gpu_etc_matrix[(int_aux2 * tasks_count) + raux1]; // Sumo el ETC de x en b.

		// Obtengo la diferencia en la máquina B.
		if (delta > current_makespan) {
			float_aux2 = delta - current_makespan;
		} else if ((float_aux2 + 1 >= current_makespan) && (delta < current_makespan)) { // sumo 1 por problemas de redondeo... funciona? 
			float_aux2 = delta - current_makespan;	
		} else {
			float_aux2 = 0.0;
		}

		// Calculo la mejora de ambos movimientos combinados.
		if ((float_aux1 != 0.0) && (float_aux2 != 0.0)) {
			if (float_aux1 > float_aux2) {
				delta = float_aux1;
			} else {
				delta = float_aux2;
			}
		} else if ((float_aux1 != 0.0) && (float_aux2 == 0.0)) {
			delta = float_aux1;
		} else if ((float_aux1 == 0.0) && (float_aux2 != 0.0)) {
			delta = float_aux2;
		} else {
			delta = 0.0;
		}

		block_swaps[thread_idx] = (ushort)((PALS_GPU_RTASK_MOVE * PALS_GPU_RTASK__THREADS) + thread_idx);
		block_swaps_delta[thread_idx] = delta;*/
	}
	
	__syncthreads();

	// Aplico reduce para quedarme con el mejor delta.
	for (int i = 1; i < PALS_GPU_RTASK__THREADS; i *= 2) {
		int pos;
		pos = 2 * i * thread_idx;
	
		if (pos < PALS_GPU_RTASK__THREADS) {
			if (block_swaps_delta[pos] > block_swaps_delta[pos + i]) {
				block_swaps_delta[pos] = block_swaps_delta[pos + i];
				block_swaps[pos] = block_swaps[pos + i];
			}
		}
	
		__syncthreads();
	}
	
	if (thread_idx == 0) {
		gpu_best_swaps[block_idx] = block_swaps[0]; //best_swap;
		gpu_best_swaps_delta[block_idx] = block_swaps_delta[0]; //best_swap_delta;
	}
}

void pals_gpu_rtask_init(struct matrix *etc_matrix, struct solution *s, 
	struct pals_gpu_rtask_instance &instance, struct pals_gpu_rtask_result &result) {
	
	// Asignación del paralelismo del algoritmo.
	instance.number_of_blocks = PALS_GPU_RTASK__BLOCKS;
	instance.threads_per_block = PALS_GPU_RTASK__THREADS;
	
	// Cantidad total de movimientos a evaluar.
	instance.total_tasks = PALS_GPU_RTASK__BLOCKS * PALS_GPU_RTASK__THREADS;
	
	if (DEBUG) {
		fprintf(stdout, "[INFO] Number of blocks (grid size)   : %d\n", instance.number_of_blocks);
		fprintf(stdout, "[INFO] Threads per block (block size) : %d\n", instance.threads_per_block);	
		fprintf(stdout, "[INFO] Total tasks                    : %d\n", instance.total_tasks);
	}

	// =========================================================================

	// Pedido de memoria en el dispositivo y copiado de datos.
	timespec ts_1;
	timming_start(ts_1);
	
	// Pido memoria para guardar el resultado.
	int best_swaps_size = sizeof(ushort) * instance.number_of_blocks;	
	if (cudaMalloc((void**)&(instance.gpu_best_swaps), best_swaps_size) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Solicitando memoria gpu_best_swaps (%d bytes).\n", best_swaps_size);
		exit(EXIT_FAILURE);
	}
		
	int best_swaps_delta_size = sizeof(float) * instance.number_of_blocks;	
	if (cudaMalloc((void**)&(instance.gpu_best_swaps_delta), best_swaps_delta_size) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Solicitando memoria gpu_best_swaps_delta (%d bytes).\n", best_swaps_delta_size);
		exit(EXIT_FAILURE);
	}
	
	timming_end(".. gpu_best_swaps", ts_1);
		
	// =========================================================================
		
	timespec ts_2;
	timming_start(ts_2);
	
	// Copio la matriz de ETC.
	int etc_matrix_size = sizeof(float) * etc_matrix->tasks_count * etc_matrix->machines_count;
	if (cudaMalloc((void**)&(instance.gpu_etc_matrix), etc_matrix_size) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Solicitando memoria etc_matrix (%d bytes).\n", etc_matrix_size);
		exit(EXIT_FAILURE);
	}
	
	if (cudaMemcpy(instance.gpu_etc_matrix, etc_matrix->data, etc_matrix_size, cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Copiando etc_matrix al dispositivo (%d bytes).\n", etc_matrix_size);
		exit(EXIT_FAILURE);
	}

	timming_end(".. gpu_etc_matrix", ts_2);

	// =========================================================================
	
	timespec ts_3;
	timming_start(ts_3);
		
	// Copio la asignación de tareas a máquinas actuales.
	int task_assignment_size = sizeof(int) * etc_matrix->tasks_count;	
	if (cudaMalloc((void**)&(instance.gpu_task_assignment), task_assignment_size) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Solicitando memoria task_assignment (%d bytes).\n", task_assignment_size);
		exit(EXIT_FAILURE);
	}
	
	if (cudaMemcpy(instance.gpu_task_assignment, s->task_assignment, task_assignment_size, cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Copiando task_assignment al dispositivo (%d bytes).\n", task_assignment_size);
		exit(EXIT_FAILURE);
	}

	timming_end(".. gpu_task_assignment", ts_3);

	// =========================================================================
	
	timespec ts_4;
	timming_start(ts_4);
		
	// Copio el compute time de las máquinas en la solución actual.
	int machine_compute_time_size = sizeof(float) * etc_matrix->machines_count;	
	if (cudaMalloc((void**)&(instance.gpu_machine_compute_time), machine_compute_time_size) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Solicitando memoria machine_compute_time (%d bytes).\n", machine_compute_time_size);
		exit(EXIT_FAILURE);
	}
	
	if (cudaMemcpy(instance.gpu_machine_compute_time, s->machine_compute_time, machine_compute_time_size, cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Copiando machine_compute_time al dispositivo (%d bytes).\n", machine_compute_time_size);
		exit(EXIT_FAILURE);
	}

	timming_end(".. gpu_machine_compute_time", ts_4);
	
	// =========================================================================
	
	if (instance.result_count > instance.number_of_blocks) instance.result_count = instance.number_of_blocks;
	
	result.move_count = instance.result_count;
	result.move_type = (char*)malloc(sizeof(char) * instance.result_count);
	result.origin = (int*)malloc(sizeof(int) * instance.result_count);
	result.destination = (int*)malloc(sizeof(int) * instance.result_count);
	result.delta = (float*)malloc(sizeof(float) * instance.result_count);
}

void pals_gpu_rtask_finalize(struct pals_gpu_rtask_instance &instance) {
	if (cudaFree(instance.gpu_etc_matrix) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Liberando la memoria solicitada para etc_matrix.\n");
		exit(EXIT_FAILURE);
	}
	
	if (cudaFree(instance.gpu_task_assignment) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Liberando la memoria solicitada para task_assignment.\n");
		exit(EXIT_FAILURE);
	}

	if (cudaFree(instance.gpu_machine_compute_time) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Liberando la memoria solicitada para machine_compute_time.\n");
		exit(EXIT_FAILURE);
	}
	
	if (cudaFree(instance.gpu_best_swaps) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Liberando la memoria solicitada para best_swaps.\n");
		exit(EXIT_FAILURE);
	}
}

void pals_gpu_rtask_clean_result(struct pals_gpu_rtask_result &result) {
	free(result.move_type);
	free(result.origin);
	free(result.destination);
	free(result.delta);
}

void pals_gpu_rtask_wrapper(struct matrix *etc_matrix, struct solution *s, 
	struct pals_gpu_rtask_instance &instance, int *gpu_random_numbers, 
	struct pals_gpu_rtask_result &result) {

	// Timming -----------------------------------------------------
	timespec ts_pals_pre;
	timming_start(ts_pals_pre);
	// Timming -----------------------------------------------------
	
	// Timming -----------------------------------------------------
	timming_end(".. pals_gpu_rtask_pals_pre", ts_pals_pre);
	// Timming -----------------------------------------------------
	
	// ==============================================================================
	// Ejecución del algoritmo.
	// ==============================================================================	
	
	// Timming -----------------------------------------------------
	timespec ts_pals;
	timming_start(ts_pals);
	// Timming -----------------------------------------------------
	
	dim3 grid(instance.number_of_blocks, 1, 1);
	dim3 threads(instance.threads_per_block, 1, 1);

	pals_rtask_kernel<<< grid, threads >>>(
		etc_matrix->machines_count,
		etc_matrix->tasks_count,
		s->makespan,
		instance.gpu_etc_matrix, 
		instance.gpu_task_assignment, 
		instance.gpu_machine_compute_time, 
		gpu_random_numbers,
		instance.gpu_best_swaps, 
		instance.gpu_best_swaps_delta);

	// Pido el espacio de memoria para obtener los resultados desde la gpu.
	ushort *best_swaps = (ushort*)malloc(sizeof(ushort) * instance.number_of_blocks);
	float *best_swaps_delta = (float*)malloc(sizeof(float) * instance.number_of_blocks);
	int *rands_nums = (int*)malloc(sizeof(int) * instance.number_of_blocks * 2);

	// Copio los mejores movimientos desde el dispositivo.
	if (cudaMemcpyAsync(best_swaps, instance.gpu_best_swaps, 
		sizeof(ushort) * instance.number_of_blocks, 
		cudaMemcpyDeviceToHost, 0) != cudaSuccess) {
		
		fprintf(stderr, "[ERROR] Copiando los mejores movimientos al host (best_swaps).\n");
		exit(EXIT_FAILURE);
	}
	
	if (cudaMemcpyAsync(best_swaps_delta, instance.gpu_best_swaps_delta, 
		sizeof(float) * instance.number_of_blocks, 
		cudaMemcpyDeviceToHost, 0) != cudaSuccess) {
		
		fprintf(stderr, "[ERROR] Copiando los mejores movimientos al host (best_swaps_delta).\n");
		exit(EXIT_FAILURE);
	}

	if (cudaMemcpyAsync(rands_nums, gpu_random_numbers, 
		sizeof(int) * instance.number_of_blocks * 2, 
		cudaMemcpyDeviceToHost, 0) != cudaSuccess) {
		
		fprintf(stderr, "[ERROR] Copiando al host los números aleatorios sorteados.\n");
		exit(EXIT_FAILURE);
	}

	cudaThreadSynchronize();

	// Timming -----------------------------------------------------
	timming_end(".. pals_gpu_rtask_pals", ts_pals);
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
	for (int i = 1; i < instance.number_of_blocks; i++) {
		if (best_swaps_delta[i] < best_swaps_delta[best_block_idx]) {
			best_block_idx = i;
		}
	}
	
	int block_idx = (best_block_idx);

	// Calculo cuales fueron los elementos modificados en ese mejor movimiento.	
	int swap = best_swaps[block_idx];

	// TODO: OPTIMIZAR!!! (pasar todo como struct? intentar meter todo dentro de un mismo array?)
	int move_type = swap / PALS_GPU_RTASK__THREADS;
	int thread_idx = swap % PALS_GPU_RTASK__THREADS;

	if (move_type == PALS_GPU_RTASK_SWAP) { // Movement type: SWAP
		int task_x = rands_nums[block_idx] % etc_matrix->tasks_count;

		int random_2 = rands_nums[block_idx + 1];
		int task_y = random_2 % (etc_matrix->tasks_count - 1 - PALS_GPU_RTASK__THREADS);
		task_y = task_y + thread_idx;

		if (task_y >= task_x) task_y = task_y + 1;
		if (task_y >= etc_matrix->tasks_count) task_y = task_y % etc_matrix->tasks_count;

		result.move_type[0] = move_type; // SWAP
		result.origin[0] = task_x;
		result.destination[0] = task_y;
		result.delta[0] = best_swaps_delta[block_idx];
		
		// =======> DEBUG
		if (DEBUG) { 
			int machine_a = s->task_assignment[task_x];
			int machine_b = s->task_assignment[task_y];

			fprintf(stdout, "[DEBUG] Task %d in %d swaps with task %d in %d. Delta %f.\n",
				task_x, machine_a, task_y, machine_b, best_swaps_delta[block_idx]);
		}
		// <======= DEBUG
	} else if (move_type == PALS_GPU_RTASK_MOVE) { // Movement type: MOVE
		int random_1 = rands_nums[block_idx] % etc_matrix->tasks_count;
		int task_x = random_1;
		int machine_a = s->task_assignment[task_x];

		int random_2 = rands_nums[block_idx + 1];
		int machine_b = (random_2 % (etc_matrix->machines_count - 1 - PALS_GPU_RTASK__THREADS)) + thread_idx;
		
		if (machine_b >= machine_a) machine_b = machine_b + 1;
		if (machine_b >= etc_matrix->machines_count) machine_b = machine_b % etc_matrix->machines_count;

		result.move_type[0] = move_type; // MOVE
		result.origin[0] = task_x;
		result.destination[0] = machine_b;
		result.delta[0] = best_swaps_delta[block_idx];
		
		// =======> DEBUG
		if (DEBUG) {
			fprintf(stdout, "[DEBUG] Task %d in %d is moved to machine %d. Delta %f.\n",
				task_x, machine_a, machine_b, best_swaps_delta[block_idx]);
		}
		// <======= DEBUG
	}
	
	// Timming -----------------------------------------------------
	timming_end(".. pals_gpu_rtask_pals_post", ts_pals_post);
	// Timming -----------------------------------------------------
}

void pals_gpu_rtask_move(struct pals_gpu_rtask_instance &instance, int task, int to_machine) {
	if (cudaMemcpy(&(instance.gpu_task_assignment[task]), &to_machine, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Error moviendo la task %d a la máquina %d.\n", task, to_machine);
		exit(EXIT_FAILURE);
	}
}

void pals_gpu_rtask_update_machine(struct pals_gpu_rtask_instance &instance, int machine, float compute_time) {
	if (cudaMemcpy(&(instance.gpu_machine_compute_time[machine]), &compute_time, sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Error actualizando el compute time de la máquina %d.\n", machine);
		exit(EXIT_FAILURE);
	}
}
