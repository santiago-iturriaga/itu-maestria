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

__global__ void pals_prtask_kernel(int machines_count, int tasks_count, float *gpu_etc_matrix, 
	int *gpu_task_assignment, float *gpu_machine_compute_time, int *gpu_random_numbers) {
	
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

				if (machine_b_ct_new > machine_a_ct_new) {
					delta = machine_b_ct_new;
				} else {
					delta = machine_a_ct_new;
				}

				if (machine_b_ct_old > machine_a_ct_old) {
					delta = delta - machine_b_ct_old;
				} else {
					delta = delta - machine_a_ct_old;
				}

				/*
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
				*/
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

			if (machine_b_ct_new > machine_a_ct_new) {
				delta = machine_b_ct_new;
			} else {
				delta = machine_a_ct_new;
			}

			if (machine_b_ct_old > machine_a_ct_old) {
				delta = delta - machine_b_ct_old;
			} else {
				delta = delta - machine_a_ct_old;
			}

			/*
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
			*/
			
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
		/*
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
		*/
	}
}

void pals_gpu_prtask_init(struct matrix *etc_matrix, struct solution *s, struct pals_gpu_prtask_instance &instance) {
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
		fprintf(stdout, "[INFO] Total tasks                    : %ld\n", instance.total_tasks);
	}
	
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
	int task_assignment_size = sizeof(int) * etc_matrix->tasks_count * PALS_GPU_PRTASK__BLOCKS;	
	
	if (cudaMalloc((void**)&(instance.gpu_task_assignment), task_assignment_size) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Solicitando memoria task_assignment (%d bytes).\n", task_assignment_size);
		exit(EXIT_FAILURE);
	} else {
		if (DEBUG) fprintf(stdout, "[DEBUG] Se solicitaron %d bytes de memoria para task_assignment.\n", task_assignment_size);
	}
	
	if (DEBUG) fprintf(stdout, "[DEBUG] task_assignment_size = %d\n", task_assignment_size);
	int *aux_host_task_assignment = (int*)(malloc(task_assignment_size));

	for (int i = 0; i < PALS_GPU_PRTASK__BLOCKS; i++) {
		int aux_size;
		aux_size = sizeof(int) * etc_matrix->tasks_count;

		int offset;
		offset = i * etc_matrix->tasks_count;
	
		if (DEBUG) fprintf(stdout, "[DEBUG] size en paso %d = %d. from %d\n", i, aux_size, offset);

		if (!memcpy(&(aux_host_task_assignment[offset]), s->task_assignment, aux_size)) {
			fprintf(stdout, "[ERROR] Copiando task_assignment\n");
			exit(EXIT_FAILURE);
		}
	}
	
	// Copio la asignación de tareas de la primer solución desde el HUESPED al DISPOSITIVO.
	if (cudaMemcpy(instance.gpu_task_assignment, aux_host_task_assignment, task_assignment_size, 
		cudaMemcpyHostToDevice) != cudaSuccess) {
		
		fprintf(stderr, "[ERROR] Copiando task_assignment al dispositivo (%d bytes).\n", task_assignment_size);
		exit(EXIT_FAILURE);
	}

	free(aux_host_task_assignment);

	timming_end(".. gpu_task_assignment", ts_3);

	// =========================================================================
	
	timespec ts_4;
	timming_start(ts_4);
		
	// Copio el compute time de las máquinas en la solución actual.
	int machine_compute_time_size = sizeof(float) * etc_matrix->machines_count * PALS_GPU_PRTASK__BLOCKS;
	
	if (cudaMalloc((void**)&(instance.gpu_machine_compute_time), machine_compute_time_size) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Solicitando memoria machine_compute_time (%d bytes).\n", machine_compute_time_size);
		exit(EXIT_FAILURE);
	}
	
	if (DEBUG) fprintf(stdout, "[DEBUG] machine_compute_time_size = %d\n", machine_compute_time_size);
	float *aux_host_machine_compute_time = (float*)(malloc(machine_compute_time_size));

	for (int i = 0; i < PALS_GPU_PRTASK__BLOCKS; i++) {
                int aux_size;
                aux_size = sizeof(float) * etc_matrix->machines_count;

                int offset;
                offset = i * etc_matrix->machines_count;

                if (DEBUG) fprintf(stdout, "[DEBUG] size en paso %d = %d. from %d\n", i, aux_size, offset);

		if (!memcpy(&(aux_host_machine_compute_time[offset]), s->machine_compute_time, aux_size)) {
			fprintf(stdout, "[ERROR] Copiando machine_compute_time\n");
			exit(EXIT_FAILURE);
		}
	}

	if (cudaMemcpy(instance.gpu_machine_compute_time, aux_host_machine_compute_time, machine_compute_time_size, 
		cudaMemcpyHostToDevice) != cudaSuccess) {
		
		fprintf(stderr, "[ERROR] Copiando machine_compute_time al dispositivo (%d bytes).\n", machine_compute_time_size);
		exit(EXIT_FAILURE);
	}

	free(aux_host_machine_compute_time);

	timming_end(".. gpu_machine_compute_time", ts_4);
	
	// =========================================================================
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
}

void pals_gpu_prtask_wrapper(struct matrix *etc_matrix, struct solution *s, 
	struct pals_gpu_prtask_instance &instance, int *gpu_random_numbers) {
	
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
		instance.gpu_etc_matrix, 
		instance.gpu_task_assignment, 
		instance.gpu_machine_compute_time, 
		gpu_random_numbers);

	if (TIMMING) cudaThreadSynchronize();

	// Timming -----------------------------------------------------
	timming_end(".. pals_gpu_prtask_pals", ts_pals);
	// Timming -----------------------------------------------------
}

void pals_gpu_prtask_get_solutions(struct matrix *etc_matrix, struct pals_gpu_prtask_instance &instance,
	int *task_assignment, float *machine_compute_time) {

	// Timming -----------------------------------------------------
	timespec ts_get;
	timming_start(ts_get);
	// Timming -----------------------------------------------------

	int machine_compute_time_size = sizeof(float) * etc_matrix->machines_count * PALS_GPU_PRTASK__BLOCKS;	
	if (cudaMemcpy(machine_compute_time, instance.gpu_machine_compute_time, machine_compute_time_size, cudaMemcpyDeviceToHost) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Copiando los compute time de las máquinas desde el dispositivo hacia el huesped (%d bytes).\n", machine_compute_time_size);
		exit(EXIT_FAILURE);
	}

	int task_assignment_size = sizeof(int) * etc_matrix->tasks_count * PALS_GPU_PRTASK__BLOCKS;	
	if (cudaMemcpy(task_assignment, instance.gpu_task_assignment, task_assignment_size, cudaMemcpyDeviceToHost) != cudaSuccess) {	
		fprintf(stderr, "[ERROR] Copiando la asignación de tareas desde el dispositivo hacia el huesped (%d bytes).\n", task_assignment_size);
		exit(EXIT_FAILURE);
	}

	// Timming -----------------------------------------------------
	timming_end(".. pals_gpu_rtask_get", ts_get);
	// Timming -----------------------------------------------------
}

void pals_gpu_prtask_join_solutions(struct pals_gpu_prtask_instance &instance, struct matrix *etc_matrix) {
	// Timming -----------------------------------------------------
	timespec ts_join;
	timming_start(ts_join);
	// Timming -----------------------------------------------------

	// TODO: pasar todo este procesamiento a la GPU!!!
	// Pido el espacio de memoria para obtener los resultados desde la gpu.
	float *machine_compute_time = (float*)malloc(sizeof(float) * etc_matrix->machines_count);
	int best_solution = 0;
	float best_solution_makespan = 0.0;

	for (int i = 0; i < PALS_GPU_PRTASK__BLOCKS; i++) {
		// Copio los mejores movimientos desde el dispositivo.
		if (cudaMemcpy(machine_compute_time, 
			instance.gpu_machine_compute_time + (i * sizeof(float) * etc_matrix->machines_count), 
			sizeof(float) * etc_matrix->machines_count, cudaMemcpyDeviceToHost) != cudaSuccess) {
		
			fprintf(stderr, "[ERROR] Copiando los mejores movimientos al host (best_swaps).\n");
			exit(EXIT_FAILURE);
		}
		
		float makespan;
		makespan = machine_compute_time[0];
		
		for (int j = 1; j < etc_matrix->machines_count; j++) {
			if (machine_compute_time[j] > makespan) {
				makespan = machine_compute_time[j];
			}
		}
		
		if (DEBUG) fprintf(stdout, "[DEBUG] Solution %d, makespan %f.\n", i, makespan);

		if (i == 0) {
			best_solution = 0;
			best_solution_makespan = makespan;
		} else {
			if (makespan < best_solution_makespan) {
				best_solution = i;
				best_solution_makespan = makespan;
			}
		}
	}

	for (int i = 0; i < PALS_GPU_PRTASK__BLOCKS; i++) {
		if (i != best_solution) {
			if (cudaMemcpy(
				instance.gpu_machine_compute_time + (i * sizeof(float) * etc_matrix->machines_count), 
				instance.gpu_machine_compute_time + (best_solution * sizeof(float) * etc_matrix->machines_count), 
				sizeof(float) * etc_matrix->machines_count, 
				cudaMemcpyDeviceToDevice) != cudaSuccess) {
			
				fprintf(stderr, "[ERROR] Copiando machine_compute_time al dispositivo (%ld bytes).\n", 
					sizeof(float) * etc_matrix->machines_count);
				exit(EXIT_FAILURE);
			}
		}
	}

	// Timming -----------------------------------------------------
	timming_end(".. pals_gpu_rtask_join", ts_join);
	// Timming -----------------------------------------------------
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
	
	pals_gpu_prtask_init(etc_matrix, current_solution, instance);

	// Timming -----------------------------------------------------
	timming_end(">> pals_gpu_prtask_init", ts_init);
	// Timming -----------------------------------------------------

	// ===========> DEBUG
	/*if (DEBUG) {
		validate_solution(etc_matrix, current_solution);
	}*/
	// <=========== DEBUG
	
	float makespan_inicial = current_solution->makespan;
	
	// Ejecuto GPUPALS.
	int seed = input.seed;
	
	RNG_rand48 r48;
	RNG_rand48_init(r48, PALS_PRTASK_RANDS);	// Debe ser múltiplo de 6144

	// Cantidad de números aleatorios por invocación.
	const unsigned int size = instance.blocks * (2 * instance.loops); // 2 random numbers por block x loop.
	const short cant_iter_generadas = PALS_PRTASK_RANDS / size;
	fprintf(stdout, "[INFO] Cantidad de iteraciones por generación de numeros aleatorios: %d.\n", cant_iter_generadas);
	
	for (int i = 0; i < PALS_COUNT; i++) {
		if (DEBUG) fprintf(stdout, "[INFO] Iteracion %d =====================\n", i);

		// ==============================================================================
		// Sorteo de numeros aleatorios.
		// ==============================================================================

		// Timming -----------------------------------------------------	
		timespec ts_rand;
		timming_start(ts_rand);
		// Timming -----------------------------------------------------
			
		if (i % cant_iter_generadas == 0) {
			if (DEBUG) fprintf(stdout, "[INFO] Generando %d números aleatorios...\n", PALS_PRTASK_RANDS);
			RNG_rand48_generate(r48, seed);
		}
	
		timming_end(">> RNG_rand48", ts_rand);

		// ==============================================================================
		// PALS.
		// ==============================================================================
	
		// Timming -----------------------------------------------------
		timespec ts_wrapper;
		timming_start(ts_wrapper);
		// Timming -----------------------------------------------------

		pals_gpu_prtask_wrapper(etc_matrix, current_solution, instance, 
			&(r48.res[(i % cant_iter_generadas) * size]));

		// Timming -----------------------------------------------------
		timming_end(">> pals_gpu_prtask_wrapper", ts_wrapper);
		// Timming -----------------------------------------------------

		// ==============================================================================
		// Punto de sincronización.
		// ==============================================================================

		// Timming -----------------------------------------------------
		timespec ts_post;
		timming_start(ts_post);
		// Timming -----------------------------------------------------

		//pals_gpu_prtask_join_solutions(instance);

		// Timming -----------------------------------------------------
		timming_end(">> pals_gpu_prtask_post", ts_post);
		// Timming -----------------------------------------------------

		// Nuevo seed.		
		seed++;
	}
	
	// Timming -----------------------------------------------------
	timespec ts_finalize;
	timming_start(ts_finalize);
	// Timming -----------------------------------------------------
	
	// Libera la memoria del dispositivo con los números aleatorios.
	RNG_rand48_cleanup(r48);
	
	// ==============================================================================
	// Obtengo las soluciones desde el dispositivo.
	// ==============================================================================
	
	int machine_compute_time_size = sizeof(float) * etc_matrix->machines_count * PALS_GPU_PRTASK__BLOCKS;	
	float *machine_compute_time;
	
	if (DEBUG) fprintf(stdout, "[DEBUG] machine_compute_time_size = %d.\n", machine_compute_time_size);

	if (!(machine_compute_time = (float*)malloc(machine_compute_time_size))) {
		fprintf(stderr, "[ERROR] Solicitando memoria para los compute time de las máquinas (%d bytes).\n", machine_compute_time_size);
		exit(EXIT_FAILURE);
	}

	int task_assignment_size = sizeof(int) * etc_matrix->tasks_count * PALS_GPU_PRTASK__BLOCKS;
	int *task_assignment;

	if (DEBUG) fprintf(stdout, "[DEBUG] task_assignment_size = %d.\n", task_assignment_size);
	
	if (!(task_assignment = (int*)malloc(task_assignment_size))) {
		fprintf(stderr, "[ERROR] Solicitando memoria para la asignación de tarea (%d bytes).\n", task_assignment_size);
		exit(EXIT_FAILURE);
	}
	
	pals_gpu_prtask_get_solutions(etc_matrix, instance, task_assignment, machine_compute_time);
	
	// ==============================================================================
	// Actualizo la solución del host con la mejor del dispositivo.
	// ==============================================================================

	int machine_compute_time_offset;
	
	int best_solution = 0;
	float best_solution_makespan = 0.0;

	for (int i = 0; i < PALS_GPU_PRTASK__BLOCKS; i++) {
		machine_compute_time_offset = i * etc_matrix->machines_count;
	
		float makespan;
		makespan = machine_compute_time[machine_compute_time_offset + 0];
		
		for (int j = 1; j < etc_matrix->machines_count; j++) {
			if (machine_compute_time[machine_compute_time_offset + j] > makespan) {
				makespan = machine_compute_time[machine_compute_time_offset + j];
			}
		}
	
		if (DEBUG) fprintf(stdout, "[DEBUG] Solution %d, makespan %f.\n", i, makespan);
	
		if (i == 0) {
			best_solution = 0;
			best_solution_makespan = makespan;
		} else {
			if (makespan < best_solution_makespan) {
				best_solution = i;
				best_solution_makespan = makespan;
			}
		}
	}
	
	if (DEBUG) fprintf(stdout, "[DEBUG] best_solution = %d.\n", best_solution);

	memcpy(current_solution->task_assignment, &(task_assignment[best_solution * etc_matrix->tasks_count]), etc_matrix->tasks_count * sizeof(int));
	memcpy(current_solution->machine_compute_time, &(machine_compute_time[best_solution * etc_matrix->machines_count]), etc_matrix->machines_count * sizeof(float));
	current_solution->makespan = best_solution_makespan;

	free(task_assignment);
	free(machine_compute_time);

	if (DEBUG) {
		for (int i = 0; i < etc_matrix->tasks_count; i++) {
			fprintf(stdout, "[DEBUG] task %d on machine %d.\n", i, current_solution->task_assignment[i]);
		}

		for (int i = 0; i < etc_matrix->machines_count; i++) {
			fprintf(stdout, "[DEBUG] machine %d compute time %f.\n", i, current_solution->machine_compute_time[i]);
		}
	}

	/*
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
	*/

	// Reconstruye el compute time de cada máquina.
	// NOTA: tengo que hacer esto cada tanto por errores acumulados en el redondeo.
	/*for (int i = 0; i < etc_matrix->machines_count; i++) {
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
	}*/
	
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

