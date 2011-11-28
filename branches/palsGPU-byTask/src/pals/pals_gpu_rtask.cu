#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <limits.h>
#include <assert.h>

#include "../config.h"
#include "../utils.h"
#include "../random/RNG_rand48.h"

#include "pals_gpu_rtask.h"

// ==============================================================================================
// NOTA. Debido al uso del generador de números aleatorios RNG_rand48:
// PALS_GPU_RTASK__BLOCKS * PALS_GPU_RTASK__LOOPS_PER_THREAD = debe ser mútiplo de 6144 (1024*6).
// ==============================================================================================
//#define PALS_GPU_RTASK__BLOCKS 		1024
//#define PALS_GPU_RTASK__THREADS		256
//#define PALS_GPU_RTASK__LOOPS_PER_THREAD 	24

//#define PALS_GPU_RTASK__BLOCKS 		2048
//#define PALS_GPU_RTASK__THREADS 		128
//#define PALS_GPU_RTASK__LOOPS_PER_THREAD 	24

#define PALS_GPU_RTASK__BLOCKS 			1024
#define PALS_GPU_RTASK__THREADS 		128
#define PALS_GPU_RTASK__LOOPS_PER_THREAD 	64

#define PALS_GPU_RTASK__MOV_TYPE_OFFSET		PALS_GPU_RTASK__THREADS * PALS_GPU_RTASK__LOOPS_PER_THREAD

__global__ void pals_rtask_kernel(int machines_count, int tasks_count, int tasks_per_thread, float *gpu_etc_matrix, 
	int *gpu_task_assignment, int *gpu_random_numbers, ushort *gpu_best_swaps, float *gpu_best_swaps_delta);	
	

void pals_gpu_rtask_init(struct matrix *etc_matrix, struct solution *s, struct pals_gpu_rtask_instance *instance) {	
	// Asignación del paralelismo del algoritmo.
	instance->number_of_blocks = PALS_GPU_RTASK__BLOCKS;
	instance->threads_per_block = PALS_GPU_RTASK__THREADS;
	instance->tasks_per_thread = PALS_GPU_RTASK__LOOPS_PER_THREAD;
	
	// Cantidad total de movimientos a evaluar.
	instance->total_tasks = PALS_GPU_RTASK__BLOCKS * PALS_GPU_RTASK__THREADS * PALS_GPU_RTASK__LOOPS_PER_THREAD;
	
	if (DEBUG) {
		fprintf(stdout, "[INFO] Number of blocks (grid size)   : %d\n", instance->number_of_blocks);
		fprintf(stdout, "[INFO] Threads per block (block size) : %d\n", instance->threads_per_block);	
		fprintf(stdout, "[INFO] Tasks per thread               : %d\n", instance->tasks_per_thread);
		fprintf(stdout, "[INFO] Total tasks                    : %d\n", instance->total_tasks);
	}

	// Pedido de memoria en el dispositivo y copiado de datos.
	timespec ts_4;
	timming_start(ts_4);
	
	// Pido memoria para guardar el resultado.
	int best_swaps_size = sizeof(ushort) * instance->number_of_blocks;	
	if (cudaMalloc((void**)&(instance->gpu_best_swaps), best_swaps_size) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Solicitando memoria gpu_best_swaps (%d bytes).\n", best_swaps_size);
		exit(EXIT_FAILURE);
	}
		
	int best_swaps_delta_size = sizeof(float) * instance->number_of_blocks;	
	if (cudaMalloc((void**)&(instance->gpu_best_swaps_delta), best_swaps_delta_size) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Solicitando memoria gpu_best_swaps_delta (%d bytes).\n", best_swaps_delta_size);
		exit(EXIT_FAILURE);
	}
	
	timming_end(".. gpu_best_swaps", ts_4);
		
	timespec ts_2;
	timming_start(ts_2);
	
	// Copio la matriz de ETC.
	int etc_matrix_size = sizeof(float) * etc_matrix->tasks_count * etc_matrix->machines_count;
	if (cudaMalloc((void**)&(instance->gpu_etc_matrix), etc_matrix_size) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Solicitando memoria etc_matrix (%d bytes).\n", etc_matrix_size);
		exit(EXIT_FAILURE);
	}
	
	if (cudaMemcpy(instance->gpu_etc_matrix, etc_matrix->data, etc_matrix_size, cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Copiando etc_matrix al dispositivo (%d bytes).\n", etc_matrix_size);
		exit(EXIT_FAILURE);
	}

	timming_end(".. gpu_etc_matrix", ts_2);

	timespec ts_3;
	timming_start(ts_3);
		
	// Copio la asignación de tareas a máquinas actuales.
	int task_assignment_size = sizeof(int) * etc_matrix->tasks_count;	
	if (cudaMalloc((void**)&(instance->gpu_task_assignment), task_assignment_size) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Solicitando memoria task_assignment (%d bytes).\n", task_assignment_size);
		exit(EXIT_FAILURE);
	}
	
	if (cudaMemcpy(instance->gpu_task_assignment, s->task_assignment, task_assignment_size, cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Copiando task_assignment al dispositivo (%d bytes).\n", task_assignment_size);
		exit(EXIT_FAILURE);
	}

	timming_end(".. gpu_task_assignment", ts_3);
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
	struct pals_gpu_rtask_instance &instance, int seed, 
	struct pals_gpu_rtask_result &result) {
	
	// ==============================================================================
	// Sorteo de numeros aleatorios.
	// ==============================================================================
	
	timespec ts_rand;
	timming_start(ts_rand);
	
	// Evals 49.152 rands => 6.291.456 movimientos (1024*24*256)(debe ser múltiplo de 6144).
	const unsigned int size = PALS_GPU_RTASK__BLOCKS * PALS_GPU_RTASK__LOOPS_PER_THREAD * 2;
	
	fprintf(stdout, "[INFO] Generando %d números aleatorios...\n", size);
	
	RNG_rand48 r48;
	RNG_rand48_init(r48, seed, size);	
	RNG_rand48_generate(r48);
	
	timming_end(".. RNG_rand48", ts_rand);
	
	// ==============================================================================
	// Ejecución del algoritmo.
	// ==============================================================================	
	
	dim3 grid(instance.number_of_blocks, 1, 1);
	dim3 threads(instance.threads_per_block, 1, 1);

	pals_rtask_kernel<<< grid, threads >>>(
		etc_matrix->machines_count,
		etc_matrix->tasks_count,
		instance.tasks_per_thread, 
		instance.gpu_etc_matrix, 
		instance.gpu_task_assignment, 
		r48.res,
		instance.gpu_best_swaps, 
		instance.gpu_best_swaps_delta);

	// Pido el espacio de memoria para obtener los resultados desde la gpu.
	ushort *best_swaps = (ushort*)malloc(sizeof(ushort) * instance.number_of_blocks);
	float *best_swaps_delta = (float*)malloc(sizeof(float) * instance.number_of_blocks);
	int *rands_nums = (int*)malloc(sizeof(int) * size);

	// Copio los mejores movimientos desde el dispositivo.
	if (cudaMemcpy(best_swaps, instance.gpu_best_swaps, sizeof(ushort) * instance.number_of_blocks, cudaMemcpyDeviceToHost) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Copiando los mejores movimientos al host (best_swaps).\n");
		exit(EXIT_FAILURE);
	}
	
	if (cudaMemcpy(best_swaps_delta, instance.gpu_best_swaps_delta, sizeof(float) * instance.number_of_blocks, cudaMemcpyDeviceToHost) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Copiando los mejores movimientos al host (best_swaps_delta).\n");
		exit(EXIT_FAILURE);
	}

	if (cudaMemcpy(rands_nums, r48.res, sizeof(int) * size, cudaMemcpyDeviceToHost) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Copiando al host los números aleatorios sorteados.\n");
		exit(EXIT_FAILURE);
	}

	// =====================================================================
	// Se cargan los resultados a la respuesta.
	// (lo mejor sería usar la GPU para generar el resultado).
	// =====================================================================
	
	if (instance.result_count > instance.number_of_blocks) instance.result_count = instance.number_of_blocks;
	
	result.move_count = instance.result_count;
	result.move_type = (char*)malloc(sizeof(char) * instance.result_count);
	result.origin = (int*)malloc(sizeof(int) * instance.result_count);
	result.destination = (int*)malloc(sizeof(int) * instance.result_count);
	result.delta = (float*)malloc(sizeof(float) * instance.result_count);

	// Busco el block que encontró el mejor movimiento.
	int best_block_idx = 0;
	for (int i = 1; i < instance.number_of_blocks; i++) {
		if (best_swaps_delta[i] < best_swaps_delta[best_block_idx]) {
			best_block_idx = i;
		}
	}
	
	for (int i = 0; i < instance.result_count; i++) {
		int block_idx = (best_block_idx + i) % instance.number_of_blocks;
	
		// Calculo cuales fueron los elementos modificados en ese mejor movimiento.	
		int mov_type_offset = PALS_GPU_RTASK__MOV_TYPE_OFFSET;
		int swap = best_swaps[block_idx];

		int move_type = swap / mov_type_offset;
		int move_offset = swap % mov_type_offset;
	
		int thread_idx = move_offset % PALS_GPU_RTASK__THREADS;
		int loop_idx = move_offset / PALS_GPU_RTASK__THREADS;

		int r_block_offset_start = block_idx * (2 * PALS_GPU_RTASK__LOOPS_PER_THREAD);

		if (move_type == PALS_GPU_RTASK_SWAP) { // Movement type: SWAP
			int random_1 = rands_nums[r_block_offset_start + loop_idx] % etc_matrix->tasks_count;
			int task_x = random_1;
	
			int random_2 = rands_nums[r_block_offset_start + loop_idx + PALS_GPU_RTASK__BLOCKS];
			int task_y = random_2 % (etc_matrix->tasks_count - 1 - PALS_GPU_RTASK__THREADS) + thread_idx;
	
			if (task_y >= task_x) task_y = task_y + 1;
			if (task_y >= etc_matrix->tasks_count) task_y = task_y % etc_matrix->tasks_count;

			result.move_type[i] = move_type; // SWAP
			result.origin[i] = task_x;
			result.destination[i] = task_y;
			result.delta[i] = best_swaps_delta[block_idx];
			
			// =======> DEBUG
			if (DEBUG) { 
				int machine_a = s->task_assignment[task_x];
				int machine_b = s->task_assignment[task_y];

				float swap_delta = 0.0;
				swap_delta -= get_etc_value(etc_matrix, machine_a, task_x); // Resto del ETC de x en a.
				swap_delta += get_etc_value(etc_matrix, machine_a, task_y); // Sumo el ETC de y en a.
				swap_delta -= get_etc_value(etc_matrix, machine_b, task_y); // Resto el ETC de y en b.
				swap_delta += get_etc_value(etc_matrix, machine_b, task_x); // Sumo el ETC de x en b.

				fprintf(stdout, "[DEBUG] Task %d in %d swaps with task %d in %d. Delta %f (%f).\n",
					task_x, machine_a, task_y, machine_b, best_swaps_delta[block_idx], swap_delta);
			}
			// <======= DEBUG
		} else if (move_type == PALS_GPU_RTASK_MOVE) { // Movement type: MOVE
			int random_1 = rands_nums[r_block_offset_start + loop_idx] % etc_matrix->tasks_count;
			int task_x = random_1;
			int machine_a = s->task_assignment[task_x];

			int random_2 = rands_nums[r_block_offset_start + loop_idx + PALS_GPU_RTASK__BLOCKS];
			int machine_b = random_2 % (etc_matrix->machines_count - 1) + thread_idx;
	
			if (machine_b >= machine_a) machine_b = machine_b + 1;
			if (machine_b >= etc_matrix->machines_count) machine_b = machine_b % etc_matrix->machines_count;

			result.move_type[i] = move_type; // MOVE
			result.origin[i] = task_x;
			result.destination[i] = machine_b;
			result.delta[i] = best_swaps_delta[block_idx];
			
			// =======> DEBUG
			if (DEBUG) {
				float swap_delta = 0.0;
				swap_delta -= get_etc_value(etc_matrix, machine_a, task_x); // Resto del ETC de x en a.
				swap_delta += get_etc_value(etc_matrix, machine_b, task_x); // Sumo el ETC de x en b.

				fprintf(stdout, "[DEBUG] Task %d in %d is moved to machine %d. Delta %f (%f).\n",
					task_x, machine_a, machine_b, best_swaps_delta[block_idx], swap_delta);
			}
			// <======= DEBUG
		}
	}
	
	// Libera la memoria del dispositivo con los números aleatorios.
	RNG_rand48_cleanup(r48);
}

void pals_gpu_rtask_move(struct pals_gpu_rtask_instance &instance, int task, int to_machine) {
	if (cudaMemset(&(instance.gpu_task_assignment[task]), to_machine, 1) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Error moviendo la task %d a la máquina %d.\n", task, to_machine);
		exit(EXIT_FAILURE);
	}
}

__global__ void pals_rtask_kernel(int machines_count, int tasks_count, int tasks_per_thread, 
	float *gpu_etc_matrix, int *gpu_task_assignment, int *gpu_random_numbers, 
	ushort *gpu_best_swaps, float *gpu_best_swaps_delta)
{
	const unsigned int thread_idx = threadIdx.x;
	const unsigned int block_idx = blockIdx.x;
	//const unsigned int block_count = gridDim.x;
	//const unsigned int thread_count = blockDim.x;

	__shared__ ushort block_best_swap;
	__shared__ float block_best_swap_delta;
	
	__shared__ ushort block_swaps[PALS_GPU_RTASK__THREADS];
	__shared__ float block_swaps_delta[PALS_GPU_RTASK__THREADS];

	// Offset de los random numbers asignados al block (2 rand x loop).
	const int r_block_offset_start = block_idx * (2 * PALS_GPU_RTASK__LOOPS_PER_THREAD);
	
	for (int loop = 0; loop < PALS_GPU_RTASK__LOOPS_PER_THREAD; loop++) {
		// El primer rand. num. es tiempre task 1.
		int raux1, raux2, aux;
		raux1 = gpu_random_numbers[r_block_offset_start + loop];
		raux2 = gpu_random_numbers[r_block_offset_start + loop + PALS_GPU_RTASK__BLOCKS];
				
		// Tipo de movimiento.	
		if (raux1 | 0x1 == raux1) { // Comparación a nivel de bit para saber si es par o impar.
			// Si es impar... 
			// Movimiento SWAP.
			raux1 = raux1 % tasks_count;
					
			raux2 = raux2 % (tasks_count - 1 - PALS_GPU_RTASK__THREADS);
			raux2 = raux2 + thread_idx;
			
			if (raux2 >= raux1) raux2 = raux2 + 1;
			if (raux2 >= tasks_count) raux2 = raux2 % tasks_count;
			
			// Calculo el delta del swap sorteado.
			float current_swap_delta = 0.0;

			aux = gpu_task_assignment[raux1]; // Máquina a.
			current_swap_delta = current_swap_delta - gpu_etc_matrix[(aux * tasks_count) + raux1]; // Resto del ETC de x en a.
			current_swap_delta = current_swap_delta + gpu_etc_matrix[(aux * tasks_count) + raux2]; // Sumo el ETC de y en a.
	
			aux = gpu_task_assignment[raux2]; // Máquina b.	
			current_swap_delta = current_swap_delta - gpu_etc_matrix[(aux * tasks_count) + raux2]; // Resto el ETC de y en b.
			current_swap_delta = current_swap_delta + gpu_etc_matrix[(aux * tasks_count) + raux1]; // Sumo el ETC de x en b.

			block_swaps[thread_idx] = (ushort)(
				(PALS_GPU_RTASK_SWAP * PALS_GPU_RTASK__MOV_TYPE_OFFSET) +
				(loop * PALS_GPU_RTASK__THREADS) + thread_idx);
			block_swaps_delta[thread_idx] = current_swap_delta;
		} else {
			// Si es par...
			// Movimiento MOVE.
			raux1 = raux1 % tasks_count;
			aux = gpu_task_assignment[raux1]; // Máquina a.
								
			raux2 = raux2 % (machines_count - 1);
			raux2 = raux2 + thread_idx;
			
			if (raux2 >= aux) raux2 = raux2 + 1;
			if (raux2 >= machines_count) raux2 = raux2 % machines_count;
			
			// Calculo el delta del swap sorteado.
			float current_swap_delta = 0.0;
			current_swap_delta = current_swap_delta - gpu_etc_matrix[(aux * tasks_count) + raux1]; // Resto del ETC de x en a.
			current_swap_delta = current_swap_delta + gpu_etc_matrix[(raux2 * tasks_count) + raux1]; // Sumo el ETC de x en b.

			block_swaps[thread_idx] = (ushort)(
				(PALS_GPU_RTASK_MOVE * PALS_GPU_RTASK__MOV_TYPE_OFFSET) +
				(loop * PALS_GPU_RTASK__THREADS) + thread_idx);
			block_swaps_delta[thread_idx] = current_swap_delta;
		}
		
		__syncthreads(); // Sincronizo todos los threads para asegurarme que todos los 
					 	 // swaps esten copiados a la memoria compartida.
	
		// Aplico reduce para quedarme con el mejor delta.
		for (int i = 1; i < PALS_GPU_RTASK__THREADS; i *= 2) {
			aux = 2 * i * thread_idx;
		
			if (aux < PALS_GPU_RTASK__THREADS) {
				if (block_swaps_delta[aux] > block_swaps_delta[aux + i]) {
					block_swaps_delta[aux] = block_swaps_delta[aux + i];
					block_swaps[aux] = block_swaps[aux + i];
				}
			}
		
			__syncthreads();
		}
		
		if (thread_idx == 0) {
			if (loop == 0) {
				block_best_swap = block_swaps[0];
				block_best_swap_delta = block_swaps_delta[0];
			} else if (block_best_swap_delta > block_swaps_delta[0]) {
				block_best_swap = block_swaps[0];
				block_best_swap_delta = block_swaps_delta[0];
			}
		}
	}
	
	if (thread_idx == 0) {
		gpu_best_swaps[block_idx] = block_best_swap; //best_swap;
		gpu_best_swaps_delta[block_idx] = block_best_swap_delta; //best_swap_delta;
	}
}

