#include <stdio.h>
#include <cuda.h>
#include <math.h>

#include "pals.h"

void fake_pals_kernel(int block_id, int thread_id, int task_count, int machine_count, struct pals_instance instance, float current_makespan);
__global__ void pals_kernel(int task_count, int machine_count, struct pals_instance instance, float current_makespan);

void pals_init(struct matrix *etc_matrix, struct solution *s, struct pals_instance *instance) {
	// Pedido de memoria en el dispositivo y copiado de datos.
	int etc_matrix_size = sizeof(float) * etc_matrix->tasks_count * etc_matrix->machines_count;
	cudaMalloc((void**)&(instance->gpu_etc_matrix), etc_matrix_size);
	cudaMemcpy(instance->gpu_etc_matrix, etc_matrix->data, etc_matrix_size, cudaMemcpyHostToDevice);	
		
	int task_assignment_size = sizeof(int) * etc_matrix->tasks_count;	
	cudaMalloc((void**)&(instance->gpu_task_assignment), task_assignment_size);
	cudaMemcpy(instance->gpu_task_assignment, s->task_assignment, task_assignment_size, cudaMemcpyHostToDevice);	
	
	int machine_compute_time_size = sizeof(float) * etc_matrix->machines_count;
	cudaMalloc((void**)&(instance->gpu_machine_compute_time), machine_compute_time_size);
	cudaMemcpy(instance->gpu_machine_compute_time, s->machine_compute_time, machine_compute_time_size, cudaMemcpyHostToDevice);
	
	// Cantidad de hilos por bloque.
	instance->block_size = 512;
	// Cantidad de swaps evalúa cada hilo.
	instance->tasks_per_thread = 4;
	// Cantidad total de swaps a evaluar.
	instance->total_tasks = etc_matrix->tasks_count * etc_matrix->tasks_count;
	
	// Cantidad de bloques necesarios para evaluar todos los swaps.
	//int number_of_blocks = (int)(ceil(etc_matrix->tasks_count / instance->block_size) * ceil(etc_matrix->tasks_count / instance->tasks_per_thread));
	int number_of_blocks = (int)ceil((etc_matrix->tasks_count * etc_matrix->tasks_count) / (instance->block_size * instance->tasks_per_thread));
	instance->number_of_blocks = number_of_blocks;
	
	fprintf(stdout, "[INFO] Block threads   : %i\n", instance->block_size);
	fprintf(stdout, "[INFO] Tasks per thread: %i\n", instance->tasks_per_thread);
	fprintf(stdout, "[INFO] Total tasks     : %i\n", instance->total_tasks);
	fprintf(stdout, "[INFO] Number of blocks: %i\n", instance->number_of_blocks);
	
	int best_swaps_size = sizeof(int) * number_of_blocks;	
	cudaMalloc((void**)&(instance->gpu_best_swaps), best_swaps_size);
}

void pals_finalize(struct pals_instance *instance) {
	cudaFree(instance->gpu_etc_matrix);
	cudaFree(instance->gpu_task_assignment);
	cudaFree(instance->gpu_machine_compute_time);
	cudaFree(instance->gpu_best_swaps);
}

void pals_wrapper(struct matrix *etc_matrix, struct solution *s, struct pals_instance *instance) {
	dim3 grid(instance->number_of_blocks, 1, 1);
	dim3 threads(instance->block_size, 1, 1);

	/*pals_kernel<<< grid, threads >>>(
		etc_matrix->tasks_count, 
		etc_matrix->machines_count, 
		*instance,
		s->makespan);*/
		
	for (int block_id = 0; block_id < instance->number_of_blocks; block_id++) {
		fprintf(stdout, "[DEBUG] Block: %i ===============================================\n", block_id);
		
		for (int thread_id = 0; thread_id < instance->block_size; thread_id++) {
			fprintf(stdout, "[DEBUG] >>> Thread: %i\n", thread_id);
			
			fake_pals_kernel(
				block_id, 
				thread_id,
				etc_matrix->tasks_count, 
				etc_matrix->machines_count, 
				*instance,
				s->makespan);
		}
	}

	//cudaMemcpy(cpu_odata,gpu_odata,nBytes,cudaMemcpyDeviceToHost);
}

void fake_pals_kernel(int block_id, int thread_id, int task_count, int machine_count, struct pals_instance instance, float current_makespan) {
	const unsigned int thread_idx = thread_id;
	const unsigned int block_idx = block_id;
	
	int block_offset_start = instance.block_size * instance.tasks_per_thread * block_idx;
	int block_offset_end = instance.block_size * instance.tasks_per_thread * (block_idx + 1) - 1;
	
	fprintf(stdout, "[DEBUG] >>>         [rango asignado al bloque: %i-%i]\n", block_offset_start, block_offset_end);
	
	int i, current_swap;
	for (i = 0; i < instance.tasks_per_thread; i++) {
		current_swap = block_offset_start + (instance.block_size * i) + thread_id;
		fprintf(stdout, "[DEBUG] >>>         [proceso swap: %i]\n", current_swap);
		fprintf(stdout, "[DEBUG] >>>         [proceso swap x: %i]\n", (int)floor((float)current_swap / (float)task_count));
		fprintf(stdout, "[DEBUG] >>>         [proceso swap y: %i]\n", (int)fmod((float)current_swap, (float)task_count));
	}
}

__global__ void pals_kernel(int task_count, int machine_count, struct pals_instance instance, float current_makespan)
{
	const unsigned int thread_idx = threadIdx.x;
	const unsigned int block_idx = blockIdx.x;
	
	int block_offset_start = instance.block_size * instance.tasks_per_thread * block_idx;
	int block_offset_end = instance.block_size * instance.tasks_per_thread * (block_idx + 1) - 1;
	
	int i, current_swap;
	for (i = 0; i < instance.tasks_per_thread; i++) {
		current_swap = block_offset_start + (instance.tasks_per_thread * instance.block_size);
	}

	// Write data to global memory
	//idx = 0;
	/*gpu_odata[thread_idx*size + idx++] = 'H';
	gpu_odata[thread_idx*size + idx++] = 'e';
	gpu_odata[thread_idx*size + idx++] = 'l';
	gpu_odata[thread_idx*size + idx++] = 'l';
	gpu_odata[thread_idx*size + idx++] = 'o';
	gpu_odata[thread_idx*size + idx++] = ' ';
	gpu_odata[thread_idx*size + idx++] = 'W';
	gpu_odata[thread_idx*size + idx++] = 'o';
	gpu_odata[thread_idx*size + idx++] = 'r';
	gpu_odata[thread_idx*size + idx++] = 'l';
	gpu_odata[thread_idx*size + idx++] = 'd';
	gpu_odata[thread_idx*size + idx++] = ' ';
	gpu_odata[thread_idx*size + idx++] = 'F';
	gpu_odata[thread_idx*size + idx++] = 'r';
	gpu_odata[thread_idx*size + idx++] = 'o';
	gpu_odata[thread_idx*size + idx++] = 'm';
	gpu_odata[thread_idx*size + idx++] = ' ';
	gpu_odata[thread_idx*size + idx++] = 'T';
	gpu_odata[thread_idx*size + idx++] = 'h';
	gpu_odata[thread_idx*size + idx++] = 'r';
	gpu_odata[thread_idx*size + idx++] = 'e';
	gpu_odata[thread_idx*size + idx++] = 'a';
	gpu_odata[thread_idx*size + idx++] = 'd';
	gpu_odata[thread_idx*size + idx++] = ' ';

	// Convert thread id to chars
	// Determine number of places in thread idx
	not_done = 1;
	k = 10;
	n = 1;
	while(not_done == 1) {
		x = thread_idx/k;
		if (x>0) {
			k = k*10;
			n +=1;
		}
		else
			not_done = 0;
	}

	// Parse out the thread index and convert to chars
	k = k/10;
	last_num = 0;
	for(i=n;i>0;i--) {
		x = thread_idx/k-last_num;
		gpu_odata[thread_idx*size + idx++] = '0' + x;
		last_num = (thread_idx/k)*10;
		k = k/10;
	}

	gpu_odata[thread_idx*size + idx++] = ' ';
	gpu_odata[thread_idx*size + idx++] = 'i';
	gpu_odata[thread_idx*size + idx++] = 'n';
	gpu_odata[thread_idx*size + idx++] = ' ';
	gpu_odata[thread_idx*size + idx++] = 'B';
	gpu_odata[thread_idx*size + idx++] = 'l';
	gpu_odata[thread_idx*size + idx++] = 'o';
	gpu_odata[thread_idx*size + idx++] = 'c';
	gpu_odata[thread_idx*size + idx++] = 'k';
	gpu_odata[thread_idx*size + idx++] = ' ';

	// Convert block id to chars
	// Determine number of places in thread idx
	not_done = 1;
	k = 10;
	n = 1;
	while(not_done == 1) {
		x = block_idx/k;
		if (x>0) {
			k = k*10;
			n +=1;
		}
		else
			not_done = 0;
	}

	// Parse out the block index and convert to chars
	k = k/10;
	last_num = 0;
	for(i=n;i>0;i--) {
		x = block_idx/k-last_num;
		gpu_odata[thread_idx*size + idx++] = '0' + x;
		last_num = (block_idx/k)*10;
		k = k/10;
	}

	// Fill out rest of string
	for(i=idx;i<size;i++)
		gpu_odata[thread_idx*size + idx++] = ' ';*/
}
