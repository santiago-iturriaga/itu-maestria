#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <limits.h>
#include <assert.h>

#include "../config.h"
#include "../utils.h"

#include "../random/cpu_rand.h"
#include "../random/RNG_rand48.h"

#include "pals_gpu_prtask.h"

#define PALS_PRTASK_RANDS 6144*20
#define PALS_PRTASK_RESULT_COUNT 1

#define PALS_GPU_PRTASK__BLOCKS 		6
#define PALS_GPU_PRTASK__THREADS 		96
#define PALS_GPU_PRTASK__LOOPS	 		1

__global__ void pals_prtask_kernel(
	int machines_count, int tasks_count, float current_makespan, float *gpu_etc_matrix, 
	int *gpu_task_assignment, float *gpu_machine_compute_time, int *gpu_random_numbers)
{
	unsigned int thread_idx = threadIdx.x;
	unsigned int block_idx = blockIdx.x;

	__shared__ unsigned short block_op[PALS_GPU_PRTASK__THREADS];
	__shared__ unsigned short block_task_x[PALS_GPU_PRTASK__THREADS];
	__shared__ unsigned short block_task_y[PALS_GPU_PRTASK__THREADS];
	__shared__ unsigned short block_machine_a[PALS_GPU_PRTASK__THREADS];
	__shared__ unsigned short block_machine_b[PALS_GPU_PRTASK__THREADS];
	__shared__ float block_machine_a_ct_new[PALS_GPU_PRTASK__THREADS];
	__shared__ float block_machine_b_ct_new[PALS_GPU_PRTASK__THREADS];
	__shared__ float block_delta[PALS_GPU_PRTASK__THREADS];
	
	unsigned int machine_compute_time_offset = block_idx * machines_count;
	unsigned int task_assignment_offset = block_idx * tasks_count;
	
	for (short loop = 0; loop < PALS_GPU_PRTASK__LOOPS; loop++) {
		int random1, random2;

		random1 = gpu_random_numbers[(block_idx * PALS_GPU_PRTASK__LOOPS * 2) + (loop * 2)];
		random2 = gpu_random_numbers[(block_idx * PALS_GPU_PRTASK__LOOPS * 2) + (loop * 2) + 1];

		short mov_type = (short)((random1 & 0x1) ^ (random2 & 0x1));
	
		// Tipo de movimiento.
		if (mov_type == 0) { // Comparación a nivel de bit para saber si es par o impar.
			// Si es impar... 
			// Movimiento SWAP.
		
			int task_x, task_y;
			int machine_a, machine_b;
		
			float machine_a_ct_old, machine_b_ct_old;
			float machine_a_ct_new, machine_b_ct_new;
		
			float delta;
			delta = 0.0;
		
			// ================= Obtengo las tareas sorteadas.
			task_x = random1 % tasks_count;
				
			task_y = ((random2 >> 1) + thread_idx) % (tasks_count - 1);	
			if (task_y >= task_x) task_y++;
		
			// ================= Obtengo las máquinas a las que estan asignadas las tareas.
			machine_a = gpu_task_assignment[task_assignment_offset + task_x]; // Máquina a.	
			machine_b = gpu_task_assignment[task_assignment_offset + task_y]; // Máquina b.	

			if (machine_a != machine_b) {
				// Calculo el delta del swap sorteado.
			
				// Máquina 1.
				machine_a_ct_old = gpu_machine_compute_time[machine_compute_time_offset + machine_a];
					
				machine_a_ct_new = machine_a_ct_old;
				machine_a_ct_new = machine_a_ct_new - gpu_etc_matrix[(machine_a * tasks_count) + task_x]; // Resto del ETC de x en a.
				machine_a_ct_new = machine_a_ct_new + gpu_etc_matrix[(machine_a * tasks_count) + task_y]; // Sumo el ETC de y en a.
			
				// Máquina 2.
				machine_b_ct_old = gpu_machine_compute_time[machine_compute_time_offset + machine_b];

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

			block_op[thread_idx] = (short)PALS_GPU_PRTASK_SWAP;
			block_task_x[thread_idx] = (unsigned short)task_x;
			block_task_y[thread_idx] = (unsigned short)task_y;
			block_machine_a[thread_idx] = (unsigned short)machine_a;
			block_machine_b[thread_idx] = (unsigned short)machine_b;
			block_machine_a_ct_new[thread_idx] = machine_a_ct_new;
			block_machine_b_ct_new[thread_idx] = machine_b_ct_new;			
			block_delta[thread_idx] = delta;
		} else {
			// Si es par...
			// Movimiento MOVE.
		
			int task_x;
			int machine_a, machine_b;
		
			float machine_a_ct_old, machine_b_ct_old;
			float machine_a_ct_new, machine_b_ct_new;

			float delta;
			delta = 0.0;
		
			// ================= Obtengo la tarea sorteada, la máquina a la que esta asignada,
			// ================= y el compute time de la máquina.
			task_x = random1 % tasks_count;
			machine_a = gpu_task_assignment[task_assignment_offset + task_x]; // Máquina a.
			
			machine_a_ct_old = gpu_machine_compute_time[machine_compute_time_offset + machine_a];	
							
			// ================= Obtengo la máquina destino sorteada.
			machine_b = ((random2 >> 1) + thread_idx) % (machines_count - 1);
			if (machine_b >= machine_a) machine_b++;
		
			machine_b_ct_old = gpu_machine_compute_time[machine_compute_time_offset + machine_b];
		
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
			
			block_op[thread_idx] = (short)PALS_GPU_PRTASK_MOVE;
			block_task_x[thread_idx] = (unsigned short)task_x;
			block_task_y[thread_idx] = 0;
			block_machine_a[thread_idx] = (unsigned short)machine_a;
			block_machine_b[thread_idx] = (unsigned short)machine_b;
			block_machine_a_ct_new[thread_idx] = machine_a_ct_new;
			block_machine_b_ct_new[thread_idx] = machine_b_ct_new;			
			block_delta[thread_idx] = delta;
		}
		
		__syncthreads();

		// Aplico reduce para quedarme con el mejor movimiento.
		int pos;
		for (int i = 1; i < PALS_GPU_PRTASK__THREADS; i *= 2) {
			pos = 2 * i * thread_idx;
	
			if (pos < PALS_GPU_PRTASK__THREADS) {
				if (block_delta[pos] > block_delta[pos + i]) {			
					block_op[pos] = block_op[pos + i];
					block_task_x[pos] = block_task_x[pos + i];
					block_task_y[pos] = block_task_y[pos + i];
					block_machine_a[pos] = block_machine_a[pos + i];
					block_machine_b[pos] = block_machine_b[pos + i];
					block_machine_a_ct_new[pos] = block_machine_a_ct_new[pos + i];
					block_machine_b_ct_new[pos] = block_machine_b_ct_new[pos + i];
					block_delta[pos] = block_delta[pos + i];
				}
			}
	
			__syncthreads();
		}
		
		// Aplico el mejor movimiento encontrado en la iteración a la solución del bloque.
		if (thread_idx == 0) {
			if (block_op[0] == PALS_GPU_PRTASK_SWAP) {
				// SWAP
				gpu_task_assignment[task_assignment_offset + block_task_x[0]] = block_machine_b[0];
				gpu_task_assignment[task_assignment_offset + block_task_y[0]] = block_machine_a[0];
		
				gpu_machine_compute_time[machine_compute_time_offset + block_machine_a[0]] = block_machine_a_ct_new[0];
				gpu_machine_compute_time[machine_compute_time_offset + block_machine_b[0]] = block_machine_b_ct_new[0];
			} else {
				// MOVE
				gpu_task_assignment[task_assignment_offset + block_task_x[0]] = block_machine_b[0];
		
				gpu_machine_compute_time[machine_compute_time_offset + block_machine_a[0]] = block_machine_a_ct_new[0];
				gpu_machine_compute_time[machine_compute_time_offset + block_machine_b[0]] = block_machine_b_ct_new[0];
			}
		}
	}
}

void pals_gpu_prtask_init(struct matrix *etc_matrix, struct solution *s, 
	struct pals_gpu_prtask_instance &instance, struct pals_gpu_prtask_result &result) {
	
	// Asignación del paralelismo del algoritmo.
	instance.blocks = PALS_GPU_PRTASK__BLOCKS;
	instance.threads = PALS_GPU_PRTASK__THREADS;
	instance.loops = PALS_GPU_PRTASK__LOOPS;
	
	// Cantidad total de movimientos a evaluar.
	instance.total_tasks = PALS_GPU_PRTASK__BLOCKS * PALS_GPU_PRTASK__THREADS * PALS_GPU_PRTASK__LOOPS;
	
	if (DEBUG) {
		fprintf(stdout, "[INFO] Number of blocks (grid size)   : %d\n", instance.blocks);
		fprintf(stdout, "[INFO] Threads per block (block size) : %d\n", instance.threads);
		fprintf(stdout, "[INFO] Loops per thread               : %d\n", instance.loops);
		fprintf(stdout, "[INFO] Total tasks                    : %d\n", instance.total_tasks);
	}

	// =========================================================================

	// Pedido de memoria en el dispositivo y copiado de datos.
	timespec ts_1;
	timming_start(ts_1);
	
	// Pido memoria para guardar el resultado.
	int best_movements_size = sizeof(int) * instance.blocks * 3;
	if (cudaMalloc((void**)&(instance.gpu_best_movements), best_movements_size) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Solicitando memoria gpu_best_movements_size (%d bytes).\n", best_movements_size);
		exit(EXIT_FAILURE);
	}
		
	int best_deltas_size = sizeof(float) * instance.blocks;	
	if (cudaMalloc((void**)&(instance.gpu_best_deltas), best_deltas_size) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Solicitando memoria gpu_best_deltas (%d bytes).\n", best_deltas_size);
		exit(EXIT_FAILURE);
	}
	
	timming_end(".. gpu_best_movements", ts_1);
		
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
	
	if (instance.result_count > instance.blocks) instance.result_count = instance.blocks;
	
	result.move_count = instance.result_count;
	result.move_type = (char*)malloc(sizeof(char) * instance.result_count);
	result.origin = (int*)malloc(sizeof(int) * instance.result_count);
	result.destination = (int*)malloc(sizeof(int) * instance.result_count);
	result.delta = (float*)malloc(sizeof(float) * instance.result_count);
}

void pals_gpu_prtask_finalize(struct pals_gpu_prtask_instance &instance) {
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
	
	if (cudaFree(instance.gpu_best_deltas) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Liberando la memoria solicitada para best_swaps.\n");
		exit(EXIT_FAILURE);
	}
	
	if (cudaFree(instance.gpu_best_movements) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Liberando la memoria solicitada para best_movements.\n");
		exit(EXIT_FAILURE);
	}
}

void pals_gpu_prtask_clean_result(struct pals_gpu_prtask_result &result) {
	free(result.move_type);
	free(result.origin);
	free(result.destination);
	free(result.delta);
}

void pals_gpu_prtask_wrapper(struct matrix *etc_matrix, struct solution *s, 
	struct pals_gpu_prtask_instance &instance, int *gpu_random_numbers, 
	struct pals_gpu_prtask_result &result) {

	// Timming -----------------------------------------------------
	timespec ts_pals_pre;
	timming_start(ts_pals_pre);
	// Timming -----------------------------------------------------
	
	// Timming -----------------------------------------------------
	timming_end(".. pals_gpu_prtask_pals_pre", ts_pals_pre);
	// Timming -----------------------------------------------------
	
	// ==============================================================================
	// Ejecución del algoritmo.
	// ==============================================================================	
	
	// Timming -----------------------------------------------------
	timespec ts_pals;
	timming_start(ts_pals);
	// Timming -----------------------------------------------------
	
	dim3 grid(instance.blocks, 1, 1);
	dim3 threads(instance.threads, 1, 1);

	pals_prtask_kernel<<< grid, threads >>>(
		etc_matrix->machines_count,
		etc_matrix->tasks_count,
		s->makespan,
		instance.gpu_etc_matrix, 
		instance.gpu_task_assignment, 
		instance.gpu_machine_compute_time, 
		gpu_random_numbers);

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

	cudaThreadSynchronize();

	// Timming -----------------------------------------------------
	timming_end(".. pals_gpu_prtask_pals", ts_pals);
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
	
	for (int i = 1; i < instance.blocks; i++) {
		if (best_deltas[i] < best_deltas[best_block_idx]) {
			best_block_idx = i;
		}

		if (DEBUG) { 
			//fprintf(stdout, ".. ID=%d, eval=%f.\n", (int)best_swaps[i], best_swaps_delta[i]);
		}
	}
	
	// Calculo cuales fueron los elementos modificados en ese mejor movimiento.	
	int block_idx = best_block_idx;

	int movement_idx = block_idx * 3;
	int move_type = best_movements[movement_idx];
	int thread_idx = best_movements[movement_idx + 1];
	int loop_idx = best_movements[movement_idx + 2];

	float delta = best_deltas[block_idx];

	int random_idx = block_idx;
	int random1 = rands_nums[random_idx];
	int random2 = rands_nums[random_idx + 1];

	if (move_type == PALS_GPU_PRTASK_SWAP) { // Movement type: SWAP
		int task_x = (random1 + loop_idx) % etc_matrix->tasks_count;

	        int task_y = ((random2 >> 1) + (loop_idx * PALS_GPU_PRTASK__THREADS) + thread_idx) % (etc_matrix->tasks_count - 1);
	        if (task_y >= task_x) task_y++;

		result.move_type[0] = move_type; // SWAP
		result.origin[0] = task_x;
		result.destination[0] = task_y;
		result.delta[0] = delta;
		
		// =======> DEBUG
		if (DEBUG) { 
			int machine_a = s->task_assignment[task_x];
			int machine_b = s->task_assignment[task_y];

			fprintf(stdout, "[DEBUG] Task %d in %d swaps with task %d in %d. Delta %f.\n",
				task_x, machine_a, task_y, machine_b, delta);
		}
		// <======= DEBUG
	} else if (move_type == PALS_GPU_PRTASK_MOVE) { // Movement type: MOVE
		int task_x = (random1 + loop_idx) % etc_matrix->tasks_count;
		int machine_a = s->task_assignment[task_x];

	        int machine_b = ((random2 >> 1) + (loop_idx * PALS_GPU_PRTASK__THREADS) + thread_idx) % (etc_matrix->machines_count - 1);
	        if (machine_b >= machine_a) machine_b++;

		result.move_type[0] = move_type; // MOVE
		result.origin[0] = task_x;
		result.destination[0] = machine_b;
		result.delta[0] = delta;
		
		// =======> DEBUG
		if (DEBUG) {
			fprintf(stdout, "[DEBUG] Task %d in %d is moved to machine %d. Delta %f.\n",
				task_x, machine_a, machine_b, delta);
		}
		// <======= DEBUG
	}
	
	// Timming -----------------------------------------------------
	timming_end(".. pals_gpu_rtask_pals_post", ts_pals_post);
	// Timming -----------------------------------------------------
}

void pals_gpu_prtask_move(struct pals_gpu_prtask_instance &instance, int task, int to_machine) {
	if (cudaMemcpy(&(instance.gpu_task_assignment[task]), &to_machine, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Error moviendo la task %d a la máquina %d.\n", task, to_machine);
		exit(EXIT_FAILURE);
	}
}

void pals_gpu_prtask_update_machine(struct pals_gpu_prtask_instance &instance, int machine, float compute_time) {
	if (cudaMemcpy(&(instance.gpu_machine_compute_time[machine]), &compute_time, sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Error actualizando el compute time de la máquina %d.\n", machine);
		exit(EXIT_FAILURE);
	}
}

void pals_gpu_prtask(struct params &input, struct matrix *etc_matrix, struct solution *current_solution) {	
	// ==============================================================================
	// PALS aleatorio por tarea.
	// ==============================================================================
	
	// Timming -----------------------------------------------------
	timespec ts_init;
	timming_start(ts_init);
	// Timming -----------------------------------------------------

	struct pals_gpu_prtask_instance instance;
	struct pals_gpu_prtask_result result;
			
	// Inicializo la memoria en el dispositivo.
	instance.result_count = PALS_PRTASK_RESULT_COUNT;
	
	pals_gpu_prtask_init(etc_matrix, current_solution, instance, result);

	if (DEBUG) {
		// Validación de la memoria del dispositivo.
		fprintf(stdout, ">> VALIDANDO MEMORIA GPU\n");

		int aux_task_assignment[etc_matrix->tasks_count];
	
		if (cudaMemcpy(aux_task_assignment, instance.gpu_task_assignment, (int)(etc_matrix->tasks_count * sizeof(int)), 
			cudaMemcpyDeviceToHost) != cudaSuccess) {
			
			fprintf(stderr, "[ERROR] Copiando task_assignment al host (%d bytes).\n", (int)(etc_matrix->tasks_count * sizeof(int)));
			exit(EXIT_FAILURE);
		}

		for (int i = 0; i < etc_matrix->tasks_count; i++) {
			if (current_solution->task_assignment[i] != aux_task_assignment[i]) {
				fprintf(stdout, "[INFO] task assignment diff => task %d on host: %d, on device: %d\n",
					i, current_solution->task_assignment[i], aux_task_assignment[i]);
			}
		}

		float aux_machine_compute_time[etc_matrix->machines_count];
	
		if (cudaMemcpy(aux_machine_compute_time, instance.gpu_machine_compute_time, (int)(etc_matrix->machines_count * sizeof(float)), 
			cudaMemcpyDeviceToHost) != cudaSuccess) {
			
			fprintf(stderr, "[ERROR] Copiando machine_compute_time al host (%d bytes).\n", (int)(etc_matrix->machines_count * sizeof(float)));
			exit(EXIT_FAILURE);
		}

		for (int i = 0; i < etc_matrix->machines_count; i++) {
			if (current_solution->machine_compute_time[i] != aux_machine_compute_time[i]) {
				fprintf(stdout, "[INFO] machine CT diff => machine %d on host: %f, on device: %f\n",
					i, current_solution->machine_compute_time[i], aux_machine_compute_time[i]);
			}
		}
	}


	// Timming -----------------------------------------------------
	timming_end(">> pals_gpu_prtask_init", ts_init);
	// Timming -----------------------------------------------------

	// ===========> DEBUG
	if (DEBUG) {
		validate_solution(etc_matrix, current_solution);
	}
	// <=========== DEBUG
	
	float makespan_inicial = current_solution->makespan;
	
	// Ejecuto GPUPALS.
	int seed = input.seed;
	
	RNG_rand48 r48;
	RNG_rand48_init(r48, PALS_PRTASK_RANDS);	// Debe ser múltiplo de 6144

	/*	
	int *gpu_randoms = NULL;
	cudaMalloc((void**)&gpu_randoms, sizeof(int) * PALS_RTASK_RANDS);
	cpu_rand_init(seed);
	*/

	// Cantidad de números aleatorios por invocación.
	const unsigned int size = instance.blocks * 2; // * instance.loops; // 2 random numbers por block x loop.
	const short cant_iter_generadas = PALS_PRTASK_RANDS / size;
	fprintf(stdout, "[INFO] Cantidad de iteraciones por generación de numeros aleatorios: %d.\n", cant_iter_generadas);
	
	for (int i = 0; i < PALS_COUNT; i++) {
		if (DEBUG) fprintf(stdout, "[INFO] Iteracion %d =====================\n", i);

		// ==============================================================================
		// Sorteo de numeros aleatorios.
		// ==============================================================================
	
		timespec ts_rand;
		timming_start(ts_rand);
	
		if (i % cant_iter_generadas == 0) {
			if (DEBUG) fprintf(stdout, "[INFO] Generando %d números aleatorios...\n", PALS_PRTASK_RANDS);
			RNG_rand48_generate(r48, seed);
			//cpu_rand_generate(gpu_randoms, PALS_RTASK_RANDS);
		}
	
		timming_end(">> RNG_rand48", ts_rand);
	
		// Timming -----------------------------------------------------
		timespec ts_wrapper;
		timming_start(ts_wrapper);
		// Timming -----------------------------------------------------

		pals_gpu_prtask_wrapper(etc_matrix, current_solution, instance, 
			&(r48.res[(i % cant_iter_generadas) * size]), result);

		/*pals_gpu_rtask_wrapper(etc_matrix, current_solution, instance, 
			&(gpu_randoms[(i % cant_iter_generadas) * size]), result);*/

		// Timming -----------------------------------------------------
		timming_end(">> pals_gpu_prtask_wrapper", ts_wrapper);
		// Timming -----------------------------------------------------

		// Timming -----------------------------------------------------
		timespec ts_post;
		timming_start(ts_post);
		// Timming -----------------------------------------------------

		// Aplico el mejor movimiento.
		if (result.delta[0] != 0.0) {
			if (result.move_type[0] == PALS_GPU_PRTASK_SWAP) {
				int task_x = result.origin[0];
				int task_y = result.destination[0];
			
				int machine_a = current_solution->task_assignment[result.origin[0]];
				int machine_b = current_solution->task_assignment[result.destination[0]];
			
				if (DEBUG) {
					fprintf(stdout, ">> [pre-update]:\n");
					fprintf(stdout, "   machine_a: %d, old_machine_a_ct: %f.\n", machine_a, current_solution->machine_compute_time[machine_a]);
					fprintf(stdout, "   machine_b: %d, old_machine_b_ct: %f.\n", machine_b, current_solution->machine_compute_time[machine_b]);
				}
			
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

				// Actualizo la asignación de cada tarea en el dispositivo.
				pals_gpu_prtask_move(instance, task_x, machine_b);
				pals_gpu_prtask_move(instance, task_y, machine_a);	
				pals_gpu_prtask_update_machine(instance, machine_a, current_solution->machine_compute_time[machine_a]);
				pals_gpu_prtask_update_machine(instance, machine_b, current_solution->machine_compute_time[machine_b]);
	
				if (DEBUG) {
					fprintf(stdout, ">> [update]:\n");
					fprintf(stdout, "   task_x: %d, task_x_machine: %d.\n", task_x, machine_b);
					fprintf(stdout, "   task_y: %d, task_y_machine: %d.\n", task_y, machine_a);
					fprintf(stdout, "   machine_a: %d, machine_a_ct: %f.\n", machine_a, current_solution->machine_compute_time[machine_a]);
					fprintf(stdout, "   machine_b: %d, machine_b_ct: %f.\n", machine_b, current_solution->machine_compute_time[machine_b]);
					fprintf(stdout, "   old_makespan: %f.\n", current_solution->makespan);
				}
			} else if (result.move_type[0] == PALS_GPU_PRTASK_MOVE) {
				int task_x = result.origin[0];		
				int machine_a = current_solution->task_assignment[task_x];
			
				//int machine_a = current_solution->task_assignment[task_x];
				int machine_b = result.destination[0];
					
				if (DEBUG) {
					fprintf(stdout, ">> [pre-update]:\n");
					fprintf(stdout, "   machine_a: %d, old_machine_a_ct: %f.\n", machine_a, current_solution->machine_compute_time[machine_a]);
					fprintf(stdout, "   machine_b: %d, old_machine_b_ct: %f.\n", machine_b, current_solution->machine_compute_time[machine_b]);
				}
					
				current_solution->task_assignment[task_x] = machine_b;
					
				// Actualizo los compute time de cada máquina luego del move en el host.
				current_solution->machine_compute_time[machine_a] = 
					current_solution->machine_compute_time[machine_a] - 
					get_etc_value(etc_matrix, machine_a, task_x);

				current_solution->machine_compute_time[machine_b] = 
					current_solution->machine_compute_time[machine_b] +
					get_etc_value(etc_matrix, machine_b, task_x);
				
				// Actualizo la asignación de cada tarea en el dispositivo.
				pals_gpu_prtask_move(instance, task_x, machine_b);
				pals_gpu_prtask_update_machine(instance, machine_a, current_solution->machine_compute_time[machine_a]);
				pals_gpu_prtask_update_machine(instance, machine_b, current_solution->machine_compute_time[machine_b]);
				
				if (DEBUG) {
					fprintf(stdout, ">> [update]:\n");
					fprintf(stdout, "   task_x: %d, task_x_machine: %d.\n", task_x, machine_b);
					fprintf(stdout, "   machine_a: %d, machine_a_ct: %f.\n", machine_a, current_solution->machine_compute_time[machine_a]);
					fprintf(stdout, "   machine_b: %d, machine_b_ct: %f.\n", machine_b, current_solution->machine_compute_time[machine_b]);
					fprintf(stdout, "   old_makespan: %f.\n", current_solution->makespan);
				}
			}
		
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

			if (DEBUG) {
				fprintf(stdout, "   new_makespan: %f (machine %d).\n", current_solution->makespan, machine);
			}
		} else {
			if (DEBUG) {
				fprintf(stdout, "   current_makespan: %f.\n", current_solution->makespan);
			}
		}

		// Timming -----------------------------------------------------
		timming_end(">> pals_gpu_prtask_post", ts_post);
		// Timming -----------------------------------------------------

		// Debug ------------------------------------------------------------------------------------------
		if (DEBUG) {
			fprintf(stdout, "[DEBUG] Mejores movimientos:\n");
			for (int i = 0; i < result.move_count; i++) {
				if (result.move_type[i] == PALS_GPU_PRTASK_SWAP) {
					int machine_a = current_solution->task_assignment[result.origin[i]];
					int machine_b = current_solution->task_assignment[result.destination[i]];
			
					fprintf(stdout, "        (swap) Task %d in %d swaps with task %d in %d. Delta %f.\n",
						result.origin[i], machine_a, result.destination[i], machine_b, result.delta[i]);
				} else if (result.move_type[i] == PALS_GPU_PRTASK_MOVE) {
					int machine_a = current_solution->task_assignment[result.origin[i]];
			
					fprintf(stdout, "        (move) Task %d in %d is moved to machine %d. Delta %f.\n",
						result.origin[i], machine_a, result.destination[i], result.delta[i]);
				}
			}
		}
		// Debug ------------------------------------------------------------------------------------------

		// Nuevo seed.		
		seed++;
	}
	
	// Timming -----------------------------------------------------
	timespec ts_finalize;
	timming_start(ts_finalize);
	// Timming -----------------------------------------------------
	
	if (DEBUG) {
		// Validación de la memoria del dispositivo.
		fprintf(stdout, ">> VALIDANDO MEMORIA GPU\n");

		int aux_task_assignment[etc_matrix->tasks_count];
	
		if (cudaMemcpy(aux_task_assignment, instance.gpu_task_assignment, (int)(etc_matrix->tasks_count * sizeof(int)), 
			cudaMemcpyDeviceToHost) != cudaSuccess) {
			
			fprintf(stderr, "[ERROR] Copiando task_assignment al host (%d bytes).\n", (int)(etc_matrix->tasks_count * sizeof(int)));
			exit(EXIT_FAILURE);
		}

		for (int i = 0; i < etc_matrix->tasks_count; i++) {
			if (current_solution->task_assignment[i] != aux_task_assignment[i]) {
				fprintf(stdout, "[INFO] task assignment diff => task %d on host: %d, on device: %d\n",
					i, current_solution->task_assignment[i], aux_task_assignment[i]);
			}
		}

		float aux_machine_compute_time[etc_matrix->machines_count];
	
		if (cudaMemcpy(aux_machine_compute_time, instance.gpu_machine_compute_time, (int)(etc_matrix->machines_count * sizeof(float)), 
			cudaMemcpyDeviceToHost) != cudaSuccess) {
			
			fprintf(stderr, "[ERROR] Copiando machine_compute_time al host (%d bytes).\n", (int)(etc_matrix->machines_count * sizeof(float)));
			exit(EXIT_FAILURE);
		}

		for (int i = 0; i < etc_matrix->machines_count; i++) {
			if (current_solution->machine_compute_time[i] != aux_machine_compute_time[i]) {
				fprintf(stdout, "[INFO] machine CT diff => machine %d on host: %f, on device: %f\n",
					i, current_solution->machine_compute_time[i], aux_machine_compute_time[i]);
			}
		}
	}
	
	// Limpio el objeto resultado.
	pals_gpu_prtask_clean_result(result);
	
	// Libera la memoria del dispositivo con los números aleatorios.
	RNG_rand48_cleanup(r48);
	//cudaFree(gpu_randoms);

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
	
	//if (DEBUG) {
		fprintf(stdout, "[DEBUG] Viejo makespan: %f\n", makespan_inicial);
		fprintf(stdout, "[DEBUG] Nuevo makespan: %f\n", current_solution->makespan);
	//}

	// Libero la memoria del dispositivo.
	pals_gpu_prtask_finalize(instance);
	
	// Timming -----------------------------------------------------
	timming_end(">> pals_gpu_prtask_finalize", ts_finalize);
	// Timming -----------------------------------------------------		
}

