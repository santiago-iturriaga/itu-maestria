#include <stdio.h>
#include <cuda.h>
#include <math.h>

#include "config.h"
#include "utils.h"
#include "pals_gpu.h"

#define THREADS_PER_BLOCK 128
#define MIN_TASKS_PER_THREAD 4
#define MAX_BLOCKS 1024
//65536

void fake_pals_kernel(int block_id, int thread_id, int task_count, int machine_count, struct matrix etc, struct solution s, 
	struct pals_gpu_instance instance);

__global__ void pals_kernel(int task_count, int block_size, int tasks_per_thread, float *gpu_etc_matrix, 
	int *gpu_task_assignment, int *gpu_best_swaps, float *gpu_best_swaps_delta);

void pals_gpu_init(struct matrix *etc_matrix, struct solution *s, struct pals_gpu_instance *instance) {
	// Cantidad de hilos por bloque.
	instance->block_size = THREADS_PER_BLOCK;
	// Cantidad total de swaps a evaluar.
	instance->total_tasks = (unsigned long)etc_matrix->tasks_count * (unsigned long)etc_matrix->tasks_count;
	// TODO: En realidad la cantidad de tasks esta dada por: (n*n)-((n+1)*(n))/2.
	//       Hay que arreglar esto y arreglar la función de coordenadas.

	int tasks_per_thread = (int)ceil(instance->total_tasks / (unsigned long)(MAX_BLOCKS * THREADS_PER_BLOCK));
	if (tasks_per_thread < MIN_TASKS_PER_THREAD) {
		// Cantidad de swaps evalúa cada hilo.
		instance->tasks_per_thread = MIN_TASKS_PER_THREAD;
		
		// Cantidad de bloques necesarios para evaluar todos los swaps.
		instance->number_of_blocks = (int)ceil(instance->total_tasks / (unsigned long)(THREADS_PER_BLOCK * MIN_TASKS_PER_THREAD));
	} else {
		instance->number_of_blocks = MAX_BLOCKS;
		instance->tasks_per_thread = tasks_per_thread;
	}
	
	if (DEBUG) {
		fprintf(stdout, "[INFO] Block size (block threads)   : %d\n", instance->block_size);
		fprintf(stdout, "[INFO] Tasks per thread             : %d\n", instance->tasks_per_thread);
		fprintf(stdout, "[INFO] Total tasks                  : %ld\n", instance->total_tasks);
		fprintf(stdout, "[INFO] Number of blocks (grid size) : %d\n", instance->number_of_blocks);
	}

	// Pedido de memoria en el dispositivo y copiado de datos.
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

	timespec ts_4;
	timming_start(ts_4);
	
	// Pido memoria para guardar el resultado.
	int best_swaps_size = sizeof(int) * instance->number_of_blocks;	
	cudaMalloc((void**)&(instance->gpu_best_swaps), best_swaps_size);
		
	int best_swaps_delta_size = sizeof(float) * instance->number_of_blocks;	
	cudaMalloc((void**)&(instance->gpu_best_swaps_delta), best_swaps_delta_size);
	
	timming_end("gpu_best_swaps", ts_4);
}

void pals_gpu_finalize(struct pals_gpu_instance *instance) {
	cudaFree(instance->gpu_etc_matrix);
	cudaFree(instance->gpu_task_assignment);
	cudaFree(instance->gpu_best_swaps);
}

void pals_gpu_wrapper(struct matrix *etc_matrix, struct solution *s, struct pals_gpu_instance *instance, 
	int &best_swaps_count, int best_swaps[], float best_swaps_delta[]) {
	
	dim3 grid(instance->number_of_blocks, 1, 1);
	dim3 threads(instance->block_size, 1, 1);

	/*
	for (int block_id = 0; block_id < instance->number_of_blocks; block_id++) {
		fprintf(stdout, "[DEBUG] Block: %i ===============================================\n", block_id);
		
		for (int thread_id = 0; thread_id < instance->block_size; thread_id++) {
			fprintf(stdout, "[DEBUG] >>> Thread: %i\n", thread_id);
			
			fake_pals_kernel(
				block_id, 
				thread_id,
				etc_matrix->tasks_count, 
				etc_matrix->machines_count, 
				*etc_matrix, *s, *instance);
		}
	}
	*/

	pals_kernel<<< grid, threads >>>(
		etc_matrix->tasks_count, 
		instance->block_size, 
		instance->tasks_per_thread, 
		instance->gpu_etc_matrix, 
		instance->gpu_task_assignment, 
		instance->gpu_best_swaps, 
		instance->gpu_best_swaps_delta);

	// Copio los mejores movimientos desde el dispositivo.
	cudaMemcpy(best_swaps, instance->gpu_best_swaps, sizeof(int) * instance->number_of_blocks, cudaMemcpyDeviceToHost);
	cudaMemcpy(best_swaps_delta, instance->gpu_best_swaps_delta, sizeof(float) * instance->number_of_blocks, cudaMemcpyDeviceToHost);
	
	best_swaps_count = instance->number_of_blocks;
}

__global__ void pals_kernel(int task_count, int block_size,
	int tasks_per_thread, float *gpu_etc_matrix, int *gpu_task_assignment, 
	int *gpu_best_swaps, float *gpu_best_swaps_delta)
{
	// Configuración optima (¿?):
	// 128 threads.
	// 16 registros por thread.
	// 2K shared memory por block.

	const unsigned int thread_idx = threadIdx.x;
	const unsigned int block_idx = blockIdx.x;

	const float block_offset_start = block_size * tasks_per_thread * block_idx;

	__shared__ int block_best_swaps[THREADS_PER_BLOCK];
	__shared__ float block_best_swaps_delta[THREADS_PER_BLOCK];

	// Busco el mejor movimiento de cada hilo.
	int i, aux;
	float auxf;

	// Coordenadas del swap.
	auxf = (block_offset_start + thread_idx) / task_count;
	int current_swap_coord_x = (int)auxf;
	int current_swap_coord_y = (int)((auxf - current_swap_coord_x) * task_count);

	// El primer task_per_thread siempre debería tener un swap válido.
	// Calculo el delta de ese primer swap y lo dejo como mejor.		
	aux = gpu_task_assignment[current_swap_coord_x]; // Máquina a.

	float current_swap_delta_xa = gpu_etc_matrix[(aux * task_count) + current_swap_coord_x]; // Resto del ETC de x en a.
	float current_swap_delta_ya = gpu_etc_matrix[(aux * task_count) + current_swap_coord_y]; // Sumo el ETC de y en a.
	
	aux = gpu_task_assignment[current_swap_coord_y]; // Máquina b.
	
	float current_swap_delta_xb = gpu_etc_matrix[(aux * task_count) + current_swap_coord_x]; // Resto el ETC de y en b.
	float current_swap_delta_yb = gpu_etc_matrix[(aux * task_count) + current_swap_coord_y]; // Sumo el ETC de x en b.

	block_best_swaps[thread_idx] = tasks_per_thread * thread_idx;
	block_best_swaps_delta[thread_idx] = current_swap_delta_ya - current_swap_delta_xa + current_swap_delta_xb - current_swap_delta_yb;

	// Para todos los demás task_per_thread.
	// En caso de que task_per_thread = 1, esto nunca se ejecuta y nunca hay divergencia de código.
	for (i = 1; i < tasks_per_thread; i++) {
		auxf = (block_offset_start  + (block_size * i) + thread_idx) / task_count;
		current_swap_coord_x = (int)auxf;
		current_swap_coord_y = (int)((auxf - current_swap_coord_x) * task_count);

		// Si la cantidad de tareas no es divisible entre la cantidad de threads
		// per block, el último bloque puede tener threads sobrantes. En este
		// caso se pierde la coherencia de los threads del último bloque.
		if (current_swap_coord_x < task_count) {
	
			// Prefiero calcular cosas inutiles con tal de mantener la coherencia entre threads.
			//if ((x < y) && (machine_a != machine_b)) {

				// Calculo el delta del swap i.
				aux = gpu_task_assignment[current_swap_coord_x]; // Máquina a.
	
				current_swap_delta_xa = gpu_etc_matrix[(aux * task_count) + current_swap_coord_x]; // Resto del ETC de x en a.
				current_swap_delta_ya = gpu_etc_matrix[(aux * task_count) + current_swap_coord_y]; // Sumo el ETC de y en a.
	
				aux = gpu_task_assignment[current_swap_coord_y]; // Máquina b.
	
				current_swap_delta_xb = gpu_etc_matrix[(aux * task_count) + current_swap_coord_x]; // Resto el ETC de y en b.
				current_swap_delta_yb = gpu_etc_matrix[(aux * task_count) + current_swap_coord_y]; // Sumo el ETC de x en b.

				auxf = current_swap_delta_ya - current_swap_delta_xa + current_swap_delta_xb - current_swap_delta_yb;
	
				if (auxf < block_best_swaps_delta[thread_idx]) {
					// Si es mejor que el mejor delta que tenía hasta el momento, lo guardo.
					
					block_best_swaps[thread_idx] = (tasks_per_thread * thread_idx) + i;
					block_best_swaps_delta[thread_idx] = auxf;
				}
		
			//}
		}
	}
	
	__syncthreads(); // Sincronizo todos los threads para asegurarme que todos los 
					 // mejores swaps esten copiados a la memoria compartida.
	
	// Aplico reduce para quedarme con el mejor delta.
	for (i = 1; i < THREADS_PER_BLOCK; i *= 2) {
		aux = 2 * i * thread_idx;
		
		if (aux < THREADS_PER_BLOCK) {
			if (block_best_swaps_delta[aux] > block_best_swaps_delta[aux + i]) {
				block_best_swaps_delta[aux] = block_best_swaps_delta[aux + i];
				block_best_swaps[aux] = block_best_swaps[aux + i];
			}
		}
		
		__syncthreads();
	}

	if (thread_idx == 0) {
		gpu_best_swaps[block_idx] = block_best_swaps[0]; //best_swap;
		gpu_best_swaps_delta[block_idx] = block_best_swaps_delta[0]; //best_swap_delta;
	}
}

void fake_pals_kernel(int block_id, int thread_id, int task_count, int machine_count, struct matrix etc, 
	struct solution s, struct pals_gpu_instance instance) {
	
	const unsigned int thread_idx = thread_id;
	const unsigned int block_idx = block_id;
	
	const int block_size = instance.block_size;	
	const int tasks_per_thread = instance.tasks_per_thread;
	
	const float block_offset_start = block_size * tasks_per_thread * block_idx;

	for (int i = 0; i < instance.tasks_per_thread; i++) {
		// Coordenadas del swap.
		float auxf = (block_offset_start + (instance.block_size * i) + thread_idx) / task_count;
		int current_swap_coord_x = (int)auxf;
		int current_swap_coord_y = (int)((auxf - current_swap_coord_x) * task_count);
	
		fprintf(stdout, "[%f] %d x %d\n", (block_offset_start + (instance.block_size * i) + thread_idx), current_swap_coord_x, current_swap_coord_y);
	}

	/*	
	int current_swap = (instance.block_size * instance.tasks_per_thread * block_idx) + (instance.block_size * 0) + thread_idx;
	
	current_swap_coord_x = (int)(current_swap / task_count);
	current_swap_coord_y = (int)(current_swap % task_count);
	
	fprintf(stdout, "%d x %d\n", current_swap_coord_x, current_swap_coord_y);
	*/
	/*const int block_size = instance.block_size;	
	const int tasks_per_thread = instance.tasks_per_thread;
	const float *gpu_etc_matrix = etc.data;
	const int *gpu_task_assignment = s.task_assignment;
	
	int block_offset_start = instance.block_size * instance.tasks_per_thread * block_idx;
	int block_offset_end = instance.block_size * instance.tasks_per_thread * (block_idx + 1) - 1;
	
	block_offset_start = block_size * tasks_per_thread * block_idx;

	// Busco el mejor movimiento de cada hilo.
	int current_swap;
	int best_swap;
	float best_swap_delta;

	// Siempre debería haber al menos un task_per_thread.
	current_swap = block_offset_start + thread_idx; // i = 0
	
	// Coordenadas del swap.
	//current_swap_coord_x = (int)floor((float)current_swap / (float)task_count);
	//current_swap_coord_y = (int)fmod((float)current_swap, (float)task_count);

	// El primer task_per_thread siempre debería tener un swap válido.
	// Calculo el delta de ese primer swap y lo dejo como mejor.
	best_swap = current_swap;
	best_swap_delta = 0.0;
		
	fprintf(stdout, "[DEBUG] >>>         [task x: %d]\n", (int)floor((float)current_swap / (float)task_count));
	fprintf(stdout, "[DEBUG] >>>         [machine a: %d]\n", gpu_task_assignment[(int)floor((float)current_swap / (float)task_count)]);
	fprintf(stdout, "[DEBUG] >>>         [task y: %d]\n", (int)fmod((float)current_swap, (float)task_count));
	fprintf(stdout, "[DEBUG] >>>         [machine b: %d]\n", gpu_task_assignment[(int)fmod((float)current_swap, (float)task_count)]);
		
	int machine = gpu_task_assignment[(int)floor((float)current_swap / (float)task_count)]; // Máquina a.
	
	best_swap_delta -= gpu_etc_matrix[machine * ((int)floor((float)current_swap / (float)task_count))]; // Resto del ETC de x en a.
	best_swap_delta += gpu_etc_matrix[machine * ((int)fmod((float)current_swap, (float)task_count))];; // Sumo el ETC de y en a.
	
	machine = gpu_task_assignment[(int)fmod((float)current_swap, (float)task_count)]; // Máquina b.
	
	best_swap_delta -= gpu_etc_matrix[machine * ((int)fmod((float)current_swap, (float)task_count))]; // Resto el ETC de y en b.
	best_swap_delta += gpu_etc_matrix[machine * ((int)floor((float)current_swap / (float)task_count))]; // Sumo el ETC de x en b.

	fprintf(stdout, "[DEBUG] >>>         [rango asignado al bloque: %i-%i]\n", block_offset_start, block_offset_end);
	fprintf(stdout, "[DEBUG] >>>         [swap      : %d]\n", best_swap);
	fprintf(stdout, "[DEBUG] >>>         [swap delta: %f]\n", best_swap_delta);*/
}
