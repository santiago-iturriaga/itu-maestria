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
//#define PALS_GPU_RTASK__BLOCKS 			1024
//#define PALS_GPU_RTASK__THREADS 			256
//#define PALS_GPU_RTASK__LOOPS_PER_THREAD 	24

//#define PALS_GPU_RTASK__BLOCKS 			2048
//#define PALS_GPU_RTASK__THREADS 			128
//#define PALS_GPU_RTASK__LOOPS_PER_THREAD 	24

#define PALS_GPU_RTASK__BLOCKS 				1024
#define PALS_GPU_RTASK__THREADS 			128
#define PALS_GPU_RTASK__LOOPS_PER_THREAD 	48

#define INT_HALF_MAX						1073741823

__global__ void pals_rtask_kernel(int machines_count, int tasks_count, int number_of_blocks, 
	int threads_per_block, int tasks_per_thread, float *gpu_etc_matrix, int *gpu_task_assignment, 
	int *gpu_random_numbers, int *gpu_best_swaps, float *gpu_best_swaps_delta);

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
	int best_swaps_size = sizeof(int) * instance->number_of_blocks;	
	cudaMalloc((void**)&(instance->gpu_best_swaps), best_swaps_size);
		
	int best_swaps_delta_size = sizeof(float) * instance->number_of_blocks;	
	cudaMalloc((void**)&(instance->gpu_best_swaps_delta), best_swaps_delta_size);
	
	timming_end("gpu_best_swaps", ts_4);
		
	timespec ts_2;
	timming_start(ts_2);
	
	// Copio la matriz de ETC.
	int etc_matrix_size = sizeof(float) * etc_matrix->tasks_count * etc_matrix->machines_count;
	cudaMalloc((void**)&(instance->gpu_etc_matrix), etc_matrix_size);
	cudaMemcpy(instance->gpu_etc_matrix, etc_matrix->data, etc_matrix_size, cudaMemcpyHostToDevice);	

	timming_end("gpu_etc_matrix", ts_2);

	timespec ts_3;
	timming_start(ts_3);
		
	// Copio la asignación de tareas a máquinas actuales.
	int task_assignment_size = sizeof(int) * etc_matrix->tasks_count;	
	cudaMalloc((void**)&(instance->gpu_task_assignment), task_assignment_size);
	cudaMemcpy(instance->gpu_task_assignment, s->task_assignment, task_assignment_size, cudaMemcpyHostToDevice);	

	timming_end("gpu_task_assignment", ts_3);
}

void pals_gpu_rtask_finalize(struct pals_gpu_rtask_instance *instance) {
	cudaFree(instance->gpu_etc_matrix);
	cudaFree(instance->gpu_task_assignment);
	cudaFree(instance->gpu_best_swaps);
}

void pals_gpu_rtask_wrapper(struct matrix *etc_matrix, struct solution *s, 
	struct pals_gpu_rtask_instance *instance, int seed, int rands_nums[], 
	int best_swaps[], float best_swaps_delta[]) {
	
	// ==============================================================================
	// Sorteo de numeros aleatorios.
	// ==============================================================================
	
	// Evals 49.152 rands => 6.291.456 movimientos (1024*24*256)(debe ser múltiplo de 6144).
	const unsigned int size = PALS_GPU_RTASK__BLOCKS * PALS_GPU_RTASK__LOOPS_PER_THREAD * 2;
	
	fprintf(stdout, "[DEBUG] Generando %d números aleatorios...\n", size);
	
	RNG_rand48 r48;
	RNG_rand48_init(r48, seed, size);	
	RNG_rand48_generate(r48);
	
	// ==============================================================================
	// Ejecución del algoritmo.
	// ==============================================================================	
	
	dim3 grid(instance->number_of_blocks, 1, 1);
	dim3 threads(instance->threads_per_block, 1, 1);

	pals_rtask_kernel<<< grid, threads >>>(
		etc_matrix->machines_count,
		etc_matrix->tasks_count,
		instance->number_of_blocks, 
		instance->threads_per_block, 
		instance->tasks_per_thread, 
		instance->gpu_etc_matrix, 
		instance->gpu_task_assignment, 
		r48.res,
		instance->gpu_best_swaps, 
		instance->gpu_best_swaps_delta);

	// Copio los mejores movimientos desde el dispositivo.
	cudaMemcpy(best_swaps, instance->gpu_best_swaps, sizeof(int) * instance->number_of_blocks, cudaMemcpyDeviceToHost);
	cudaMemcpy(best_swaps_delta, instance->gpu_best_swaps_delta, sizeof(float) * instance->number_of_blocks, cudaMemcpyDeviceToHost);
	cudaMemcpy(rands_nums, r48.res, sizeof(int) * size, cudaMemcpyDeviceToHost);
	
	// Libera la memoria del dispositivo con los números aleatorios.
	RNG_rand48_cleanup(r48);
}

__global__ void pals_rtask_kernel(int machines_count, int tasks_count, int number_of_blocks, 
	int threads_per_block, int tasks_per_thread, float *gpu_etc_matrix, int *gpu_task_assignment, 
	int *gpu_random_numbers, int *gpu_best_swaps, float *gpu_best_swaps_delta)
{
	//PALS_GPU_RTASK__BLOCKS 			1024
	//PALS_GPU_RTASK__THREADS 			128
	//PALS_GPU_RTASK__LOOPS_PER_THREAD 	48

	const unsigned int thread_idx = threadIdx.x;
	const unsigned int block_idx = blockIdx.x;

	__shared__ int block_best_swap;
	__shared__ float block_best_swap_delta;
	
	__shared__ short block_swaps[PALS_GPU_RTASK__THREADS];
	__shared__ float block_swaps_delta[PALS_GPU_RTASK__THREADS];

	// Offset de los random numbers asignados al block (2 rand x loop).
	const int r_block_offset_start = block_idx * (2 * PALS_GPU_RTASK__LOOPS_PER_THREAD);
		
	for (int loop = 0; loop < PALS_GPU_RTASK__LOOPS_PER_THREAD; loop++) {
		// El primer rand. num. es tiempre task 1.
		int raux1, raux2, aux;
		raux1 = gpu_random_numbers[r_block_offset_start + loop];
	
		// Tipo de movimiento.	
		//if (raux1 % 2 == 0) { //TODO: el módulo es muy ineficiente.
			// Movimiento SWAP.
			raux1 = (int)(((float)tasks_count / (float)INT_MAX) * (float)raux1);
			//assert(raux1 < tasks_count);
			
			raux2 = gpu_random_numbers[r_block_offset_start + loop + 1];
			raux2 = (int)(((float)(tasks_count-1-PALS_GPU_RTASK__THREADS) / (float)INT_MAX) * (float)(raux2-1-PALS_GPU_RTASK__THREADS));
			raux2 = raux2 + thread_idx;
			
			if (raux2 >= raux1) {
				raux2 = raux2 + 1;
				
				if (raux2 == tasks_count) raux2 = 0;
			}
			//assert(raux2 < tasks_count);
			
			// Calculo el delta del swap sorteado.
			float current_swap_delta;

			aux = gpu_task_assignment[raux1]; // Máquina a.
			current_swap_delta = 0.0 - gpu_etc_matrix[(aux * tasks_count) + raux1]; // Resto del ETC de x en a.
			current_swap_delta = current_swap_delta + gpu_etc_matrix[(aux * tasks_count) + raux2]; // Sumo el ETC de y en a.
	
			aux = gpu_task_assignment[raux2]; // Máquina b.	
			current_swap_delta = current_swap_delta - gpu_etc_matrix[(aux * tasks_count) + raux2]; // Resto el ETC de y en b.
			current_swap_delta = current_swap_delta + gpu_etc_matrix[(aux * tasks_count) + raux1]; // Sumo el ETC de x en b.

			block_swaps[thread_idx] = (loop * PALS_GPU_RTASK__THREADS) + thread_idx;
			block_swaps_delta[thread_idx] = current_swap_delta;
		//} else {
			// Movimiento MOVE.
			// TODO: hacer!!!
		//}
		
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
			if ((loop == 0) || (block_best_swap_delta > block_swaps_delta[0])) {
				block_best_swap = block_swaps[thread_idx];
				block_best_swap_delta = block_swaps_delta[thread_idx];
			}
		}
	}
	
	if (thread_idx == 0) {
		gpu_best_swaps[block_idx] = (int)block_best_swap; //best_swap;
		gpu_best_swaps_delta[block_idx] = block_best_swap_delta; //best_swap_delta;
	}
}

