#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <limits.h>
#include <assert.h>

#include "../config.h"
#include "../utils.h"

#include "../random/cpu_rand.h"
#include "../random/RNG_rand48.h"
#include "../random/mtgp-1.1/mtgp32-cuda.h"

#include "pals_gpu_rtask.h"

#define VERY_BIG_FLOAT                  1073741824
//#define PALS_RTASK_RANDS                6144*20
#define PALS_RTASK_RANDS                1048576

#define TOTALLY_FUCKUP_AUX_SIZE         6

#define PALS_GPU_RTASK__BLOCKS          64
#define PALS_GPU_RTASK__THREADS         256
#define PALS_GPU_RTASK__LOOPS           1

#define APPLY_BEST_KERNEL_THREADS       PALS_GPU_RTASK__BLOCKS >> 1

// No puedo trabajar con más de COMPUTE_MAKESPAN_KERNEL_THREADS * COMPUTE_MAKESPAN_KERNEL_BLOCKS machines.
// 512 * 1 * 2 = 1024
#define COMPUTE_MAKESPAN_KERNEL_BLOCKS        1
#define COMPUTE_MAKESPAN_KERNEL_THREADS       512

__global__ void pals_rtask_kernel(
    int machines_count,
    int tasks_count, 
    float *gpu_etc_matrix,
    int *gpu_task_assignment, 
    float *gpu_machine_compute_time,
    int *gpu_random_numbers, 
    int *gpu_best_movements_op,
    int *gpu_best_movements_data1, 
    int *gpu_best_movements_data2,
    float *gpu_best_deltas)
{
    const unsigned int thread_idx = threadIdx.x;
    const unsigned int block_idx = blockIdx.x;
    const unsigned int block_dim = blockDim.x; // Cantidad de threads.

    const short mov_type = (short)(block_idx & 0x11);

    const unsigned int random1 = gpu_random_numbers[2 * block_idx];
    const unsigned int random2 = gpu_random_numbers[(2 * block_idx) + 1];

    __shared__ short block_op[PALS_GPU_RTASK__THREADS];
    __shared__ int block_data1[PALS_GPU_RTASK__THREADS];
    __shared__ int block_data2[PALS_GPU_RTASK__THREADS];
    __shared__ float block_deltas[PALS_GPU_RTASK__THREADS];

    for (int loop = 0; loop < PALS_GPU_RTASK__LOOPS; loop++) {
        // Tipo de movimiento.
        if (mov_type <= 2) {
            // PALS_GPU_RTASK_SWAP
            
            int task_x, task_y;
            int machine_a, machine_b;

            float machine_a_ct_old, machine_b_ct_old;
            float machine_a_ct_new, machine_b_ct_new;

            float delta;
            delta = VERY_BIG_FLOAT;

            // ================= Obtengo las tareas sorteadas.
            task_x = (random1 + loop) & (tasks_count-1);
            task_y = (random2 + (loop * block_dim) + thread_idx) & (tasks_count - 1);

            if (task_x != task_y) {
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

                    float max_old;
                    max_old = machine_a_ct_old;
                    if (max_old < machine_b_ct_old) max_old = machine_b_ct_old;
                    
                    delta = (machine_a_ct_new - max_old) + (machine_b_ct_new - max_old);
                }
            }
            if ((loop == 0) || (block_deltas[thread_idx] > delta)) {
                block_op[thread_idx] = PALS_GPU_RTASK_SWAP;
                block_data1[thread_idx] = task_x;
                block_data2[thread_idx] = task_y;
                block_deltas[thread_idx] = delta;
            }
        } else {
            // PALS_GPU_RTASK_MOVE

            int task_x;
            int machine_a, machine_b;

            float machine_a_ct_old, machine_b_ct_old;
            float machine_a_ct_new, machine_b_ct_new;

            float delta;
            delta = VERY_BIG_FLOAT;

            // ================= Obtengo la tarea sorteada, la máquina a la que esta asignada,
            // ================= y el compute time de la máquina.
            task_x = (random1 + loop) & (tasks_count - 1);
            machine_a = gpu_task_assignment[task_x]; // Máquina a.

            // ================= Obtengo la máquina destino sorteada.
            machine_b = (random2 + (loop * block_dim) + thread_idx) & (machines_count - 1);

            if (machine_a != machine_b) {
                machine_a_ct_old = gpu_machine_compute_time[machine_a];
                machine_b_ct_old = gpu_machine_compute_time[machine_b];

                // Calculo el delta del swap sorteado.
                machine_a_ct_new = machine_a_ct_old - gpu_etc_matrix[(machine_a * tasks_count) + task_x]; // Resto del ETC de x en a.
                machine_b_ct_new = machine_b_ct_old + gpu_etc_matrix[(machine_b * tasks_count) + task_x]; // Sumo el ETC de x en b.

                float max_old;
                max_old = machine_a_ct_old;
                if (max_old < machine_b_ct_old) max_old = machine_b_ct_old;
                
                delta = (machine_a_ct_new - max_old) + (machine_b_ct_new - max_old);
            }

            if ((loop == 0) || (block_deltas[thread_idx] > delta)) {                
                block_op[thread_idx] = PALS_GPU_RTASK_MOVE;
                block_data1[thread_idx] = task_x;
                block_data2[thread_idx] = machine_b;
                block_deltas[thread_idx] = delta;
            }
        }
    }

    __syncthreads();

    // Aplico reduce para quedarme con el mejor delta.
    float delta_current, delta;

    for(unsigned int s=block_dim/2; s>0; s>>=1)
    {
        if (thread_idx < s)
        {
            delta_current = block_deltas[thread_idx];
            delta = block_deltas[thread_idx + s];

            if (delta < delta_current) {
                block_op[thread_idx] = block_op[thread_idx + s];
                block_data1[thread_idx] = block_data1[thread_idx + s];
                block_data2[thread_idx] = block_data2[thread_idx + s];
                block_deltas[thread_idx] = delta;
            }
        }
        __syncthreads();
    }

    if (thread_idx == 0) {
        gpu_best_movements_op[block_idx] = block_op[0];
        gpu_best_movements_data1[block_idx] = block_data1[0];
        gpu_best_movements_data2[block_idx] = block_data2[0];
        gpu_best_deltas[block_idx] = block_deltas[0];
    }
}

__global__ void pals_apply_best_kernel(
    int machines_count,
    int tasks_count, 
    float *gpu_etc_matrix,
    int *gpu_task_assignment, 
    float *gpu_machine_compute_time,
    int *gpu_best_movements_op,
    int *gpu_best_movements_data1, 
    int *gpu_best_movements_data2,
    float *gpu_best_deltas) {

    __shared__ unsigned int sdata_idx[APPLY_BEST_KERNEL_THREADS];
    __shared__ float sdata[APPLY_BEST_KERNEL_THREADS];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x;

    float delta = (i < PALS_GPU_RTASK__BLOCKS) ? gpu_best_deltas[i] : VERY_BIG_FLOAT;
    unsigned int delta_idx = i;

    if (i + blockDim.x < PALS_GPU_RTASK__BLOCKS) {
        float aux = gpu_best_deltas[i + blockDim.x];
        if (aux < delta) {
            delta = aux;
            delta_idx = i + blockDim.x;
        }
    }

    sdata[tid] = delta;
    sdata_idx[tid] = delta_idx;

    __syncthreads();

    float delta_current;

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            delta_current = sdata[tid];
            delta = sdata[tid + s];

            if (delta < delta_current) {
                sdata[tid] = delta;
                sdata_idx[tid] = sdata_idx[tid + s];
            }
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) {
        i = sdata_idx[0];

        int mov_type = gpu_best_movements_op[i];
        int data1 = gpu_best_movements_data1[i];
        int data2 = gpu_best_movements_data2[i];
        float delta = sdata[0];

        gpu_best_movements_op[0] = mov_type;
        gpu_best_movements_data1[1] = data1;
        gpu_best_movements_data2[2] = data2;
        gpu_best_deltas[3] = delta;

        if (mov_type == PALS_GPU_RTASK_SWAP) {
            // Movimiento SWAP.

            int task_x, task_y;
            task_x = data1;
            task_y = data2;
            
            int machine_a, machine_b;
            machine_a = gpu_task_assignment[task_x]; // Máquina a.
            machine_b = gpu_task_assignment[task_y]; // Máquina b.

            // UPDATE!
            gpu_task_assignment[task_x] = machine_b;
            gpu_task_assignment[task_y] = machine_a;

            float aux_ct;
            aux_ct = gpu_machine_compute_time[machine_a];
            aux_ct = aux_ct - gpu_etc_matrix[(machine_a * tasks_count) + task_x];
            aux_ct = aux_ct + gpu_etc_matrix[(machine_a * tasks_count) + task_y];
            gpu_machine_compute_time[machine_a] = aux_ct;

            aux_ct = gpu_machine_compute_time[machine_b];
            aux_ct = aux_ct - gpu_etc_matrix[(machine_b * tasks_count) + task_y];
            aux_ct = aux_ct + gpu_etc_matrix[(machine_b * tasks_count) + task_x];
            gpu_machine_compute_time[machine_b] = aux_ct;
        } else {
            // Movimiento MOVE.

            int task_x;
            task_x = data1;
            
            int machine_a, machine_b;
            machine_a = gpu_task_assignment[task_x]; // Máquina a.
            machine_b = data2;

            // UPDATE!
            gpu_task_assignment[task_x] = machine_b;

            float aux_ct;
            aux_ct = gpu_machine_compute_time[machine_a];
            aux_ct = aux_ct - gpu_etc_matrix[(machine_a * tasks_count) + task_x];
            gpu_machine_compute_time[machine_a] = aux_ct;

            aux_ct = gpu_machine_compute_time[machine_b];
            aux_ct = aux_ct + gpu_etc_matrix[(machine_b * tasks_count) + task_x];
            gpu_machine_compute_time[machine_b] = aux_ct;
        }
    }
}

__global__ void pals_compute_makespan(
    int machines_count,
    float *gpu_machine_compute_time, 
    int *machine_idx, 
    float *machine_ct) {

    __shared__ unsigned int sdata_idx[COMPUTE_MAKESPAN_KERNEL_THREADS];
    __shared__ float sdata[COMPUTE_MAKESPAN_KERNEL_THREADS];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float ct = (i < machines_count) ? gpu_machine_compute_time[i] : 0;
    unsigned int ct_idx = i;

    if (i + blockDim.x < machines_count) {
        float aux = gpu_machine_compute_time[i + blockDim.x];
        if (aux > ct) {
            ct = aux;
            ct_idx = i + blockDim.x;
        }
    }

    sdata[tid] = ct;
    sdata_idx[tid] = ct_idx;

    __syncthreads();

    float ct_current;

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            ct_current = sdata[tid];
            ct = sdata[tid + s];

            if (ct > ct_current) {
                sdata[tid] = ct;
                sdata_idx[tid] = sdata_idx[tid + s];
            }
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) {
        machine_idx[blockIdx.x] = sdata_idx[0];
        machine_ct[blockIdx.x] = sdata[0];
    }
}

void pals_gpu_rtask_init(struct matrix *etc_matrix, struct solution *s,
    struct pals_gpu_rtask_instance &instance) {

    // Asignación del paralelismo del algoritmo.
    instance.blocks = PALS_GPU_RTASK__BLOCKS;
    instance.threads = PALS_GPU_RTASK__THREADS;
    instance.loops = PALS_GPU_RTASK__LOOPS;

    // Cantidad total de movimientos a evaluar.
    instance.total_tasks = instance.blocks * instance.threads * instance.loops;

    if (DEBUG) {
        fprintf(stdout, "[INFO] Number of blocks (grid size)   : %d\n", instance.blocks);
        fprintf(stdout, "[INFO] Threads per block (block size) : %d\n", instance.threads);
        fprintf(stdout, "[INFO] Loops per thread               : %d\n", instance.loops);
        fprintf(stdout, "[INFO] Total tasks                    : %ld\n", instance.total_tasks);
    }

    // =========================================================================

    // Pedido de memoria en el dispositivo y copiado de datos.
    timespec ts_1;
    timming_start(ts_1);

    // Pido memoria para guardar el resultado.
    int best_movements_size = sizeof(int) * instance.blocks;
    if (cudaMalloc((void**)&(instance.gpu_best_movements_op), best_movements_size) != cudaSuccess) {
        fprintf(stderr, "[ERROR] Solicitando memoria gpu_best_movements_op (%d bytes).\n", best_movements_size);
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void**)&(instance.gpu_best_movements_data1), best_movements_size) != cudaSuccess) {
        fprintf(stderr, "[ERROR] Solicitando memoria gpu_best_movements_data1 (%d bytes).\n", best_movements_size);
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void**)&(instance.gpu_best_movements_data2), best_movements_size) != cudaSuccess) {
        fprintf(stderr, "[ERROR] Solicitando memoria gpu_best_movements_data2 (%d bytes).\n", best_movements_size);
        exit(EXIT_FAILURE);
    }

    int best_deltas_size = sizeof(float) * instance.blocks;
    if (cudaMalloc((void**)&(instance.gpu_best_deltas), best_deltas_size) != cudaSuccess) {
        fprintf(stderr, "[ERROR] Solicitando memoria gpu_best_deltas (%d bytes).\n", best_deltas_size);
        exit(EXIT_FAILURE);
    }

    if (cudaMalloc((void**)&(instance.gpu_makespan_idx_aux), sizeof(int) * COMPUTE_MAKESPAN_KERNEL_BLOCKS) != cudaSuccess) {
        fprintf(stderr, "[ERROR] Solicitando memoria gpu_makespan_idx_aux (%lu bytes).\n", sizeof(int) * COMPUTE_MAKESPAN_KERNEL_BLOCKS);
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void**)&(instance.gpu_makespan_ct_aux), sizeof(int) * COMPUTE_MAKESPAN_KERNEL_BLOCKS) != cudaSuccess) {
        fprintf(stderr, "[ERROR] Solicitando memoria gpu_makespan_ct_aux (%lu bytes).\n", sizeof(int) * COMPUTE_MAKESPAN_KERNEL_BLOCKS);
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

    if (cudaFree(instance.gpu_best_deltas) != cudaSuccess) {
        fprintf(stderr, "[ERROR] Liberando la memoria solicitada para best_swaps.\n");
        exit(EXIT_FAILURE);
    }

    if (cudaFree(instance.gpu_best_movements_op) != cudaSuccess) {
        fprintf(stderr, "[ERROR] Liberando la memoria solicitada para gpu_best_movements_op.\n");
        exit(EXIT_FAILURE);
    }

    if (cudaFree(instance.gpu_best_movements_data1) != cudaSuccess) {
        fprintf(stderr, "[ERROR] Liberando la memoria solicitada para gpu_best_movements_data1.\n");
        exit(EXIT_FAILURE);
    }

    if (cudaFree(instance.gpu_best_movements_data2) != cudaSuccess) {
        fprintf(stderr, "[ERROR] Liberando la memoria solicitada para gpu_best_movements_data2.\n");
        exit(EXIT_FAILURE);
    }

    if (cudaFree(instance.gpu_makespan_idx_aux) != cudaSuccess) {
        fprintf(stderr, "[ERROR] Liberando la memoria solicitada para gpu_makespan_idx_aux.\n");
        exit(EXIT_FAILURE);
    }

    if (cudaFree(instance.gpu_makespan_ct_aux) != cudaSuccess) {
        fprintf(stderr, "[ERROR] Liberando la memoria solicitada para gpu_makespan_ct_aux.\n");
        exit(EXIT_FAILURE);
    }
}

void show_search_results(struct matrix *etc_matrix, struct solution *s,
    struct pals_gpu_rtask_instance &instance, int *gpu_random_numbers) {

    // Pido el espacio de memoria para obtener los resultados desde la gpu.
    int *best_movements_op = (int*)malloc(sizeof(int) * instance.blocks);
    int *best_movements_data1 = (int*)malloc(sizeof(int) * instance.blocks);
    int *best_movements_data2 = (int*)malloc(sizeof(int) * instance.blocks);
    float *best_deltas = (float*)malloc(sizeof(float) * instance.blocks);
    int *rands_nums = (int*)malloc(sizeof(int) * instance.blocks * 2);

    // Copio los mejores movimientos desde el dispositivo.
    if (cudaMemcpy(best_movements_op, instance.gpu_best_movements_op, sizeof(int) * instance.blocks,
        cudaMemcpyDeviceToHost) != cudaSuccess) {

        fprintf(stderr, "[ERROR] Copiando los mejores movimientos al host (gpu_best_movements_op).\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMemcpy(best_movements_data1, instance.gpu_best_movements_data1, sizeof(int) * instance.blocks,
        cudaMemcpyDeviceToHost) != cudaSuccess) {

        fprintf(stderr, "[ERROR] Copiando los mejores movimientos al host (gpu_best_movements_data1).\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMemcpy(best_movements_data2, instance.gpu_best_movements_data2, sizeof(int) * instance.blocks,
        cudaMemcpyDeviceToHost) != cudaSuccess) {

        fprintf(stderr, "[ERROR] Copiando los mejores movimientos al host (gpu_best_movements_data2).\n");
        exit(EXIT_FAILURE);
    }

    if (cudaMemcpy(best_deltas, instance.gpu_best_deltas, sizeof(float) * instance.blocks,
        cudaMemcpyDeviceToHost) != cudaSuccess) {

        fprintf(stderr, "[ERROR] Copiando los mejores movimientos al host (gpu_best_deltas).\n");
        exit(EXIT_FAILURE);
    }

    if (cudaMemcpy(rands_nums, gpu_random_numbers, sizeof(int) * instance.blocks * 2,
        cudaMemcpyDeviceToHost) != cudaSuccess) {

        fprintf(stderr, "[ERROR] Copiando al host los números aleatorios sorteados.\n");
        exit(EXIT_FAILURE);
    }
       
    for (int block_idx = 0; block_idx < instance.blocks; block_idx++) {
        // Calculo cuales fueron los elementos modificados en ese mejor movimiento. 
        int move_type = best_movements_op[block_idx];
        int data1 = best_movements_data1[block_idx];
        int data2 = best_movements_data2[block_idx];
        float delta = best_deltas[block_idx];

        unsigned int random1 = rands_nums[2 * block_idx];
        unsigned int random2 = rands_nums[(2 * block_idx) + 1];
        
        fprintf(stdout, "RANDOMS: %u %u\n", random1, random2);

        if (move_type == PALS_GPU_RTASK_SWAP) { // Movement type: SWAP
            int task_x = data1;
            int task_y = data2;
        
            // =======> DEBUG
            if (DEBUG) { 
                int machine_a = s->task_assignment[task_x];
                int machine_b = s->task_assignment[task_y];

                fprintf(stdout, "[DEBUG] Task %d in %d swaps with task %d in %d. Delta %f.\n",
                    task_x, machine_a, task_y, machine_b, delta);
            }
            // <======= DEBUG
        } else if (move_type == PALS_GPU_RTASK_MOVE) { // Movement type: MOVE
            int task_x = data1;
            int machine_a = s->task_assignment[task_x];
            int machine_b = data2;
        
            // =======> DEBUG
            if (DEBUG) {
                fprintf(stdout, "[DEBUG] Task %d in %d is moved to machine %d. Delta %f.\n",
                    task_x, machine_a, machine_b, delta);
            }
            // <======= DEBUG
        }
    }
}

void pals_gpu_rtask_wrapper(struct matrix *etc_matrix, struct solution *s,
    struct pals_gpu_rtask_instance &instance, int *gpu_random_numbers) {

    // ==============================================================================
    // Ejecución del algoritmo.
    // ==============================================================================

    // Timming -----------------------------------------------------
    timespec ts_pals;
    timming_start(ts_pals);
    // Timming -----------------------------------------------------

    if (DEBUG) cudaThreadSynchronize();

    dim3 grid(instance.blocks, 1, 1);
    dim3 threads(instance.threads, 1, 1);

    pals_rtask_kernel<<< grid, threads >>>(
        etc_matrix->machines_count,
        etc_matrix->tasks_count,
        instance.gpu_etc_matrix,
        instance.gpu_task_assignment,
        instance.gpu_machine_compute_time,
        gpu_random_numbers,
        instance.gpu_best_movements_op,
        instance.gpu_best_movements_data1,
        instance.gpu_best_movements_data2,
        instance.gpu_best_deltas);

    cudaError_t e;
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "[ERROR] Failure in kernel call.\n%s\n", cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }

    if (DEBUG) cudaThreadSynchronize();

    // Timming -----------------------------------------------------
    timming_end(".. pals_gpu_rtask_pals", ts_pals);
    // Timming -----------------------------------------------------

    if (DEBUG) show_search_results(etc_matrix, s, instance, gpu_random_numbers);

    // =====================================================================
    // Hago un reduce y aplico el mejor movimiento.
    // =====================================================================

    // Timming -----------------------------------------------------
    timespec ts_pals_reduce;
    timming_start(ts_pals_reduce);
    // Timming -----------------------------------------------------

    /*
    pals_apply_best_kernel<<< 1, APPLY_BEST_KERNEL_THREADS >>>(
        etc_matrix->machines_count,
        etc_matrix->tasks_count,
        instance.gpu_etc_matrix,
        instance.gpu_task_assignment,
        instance.gpu_machine_compute_time,
        gpu_random_numbers,
        instance.gpu_best_movements_op,
        instance.gpu_best_movements_thread,
        instance.gpu_best_movements_loop,
        instance.gpu_best_deltas,
        instance.gpu_totally_fuckup_aux);

    if (DEBUG) cudaThreadSynchronize();
    */
    // Timming -----------------------------------------------------
    timming_end(".. pals_gpu_rtask_reduce", ts_pals_reduce);
    // Timming -----------------------------------------------------

    // =====================================================================
    // Se carga el mejor de los movimientos.
    // =====================================================================

    // Timming -----------------------------------------------------
    timespec ts_pals_post;
    timming_start(ts_pals_post);
    // Timming -----------------------------------------------------

    // Timming -----------------------------------------------------
    timming_end(".. pals_gpu_rtask_pals_post", ts_pals_post);
    // Timming -----------------------------------------------------
}

void load_sol_from_gpu(struct matrix *etc_matrix, struct pals_gpu_rtask_instance &instance, struct solution *cpu_solution) {
    int task_assignment_size = sizeof(int) * etc_matrix->tasks_count;
    if (cudaMemcpy(cpu_solution->task_assignment, instance.gpu_task_assignment,
        task_assignment_size, cudaMemcpyDeviceToHost) != cudaSuccess) {

        fprintf(stderr, "[ERROR] Copiando task_assignment desde el dispositivo (%d bytes).\n", task_assignment_size);
        exit(EXIT_FAILURE);
    }

    int machine_compute_time_size = sizeof(float) * etc_matrix->machines_count;
    if (cudaMemcpy(cpu_solution->machine_compute_time, instance.gpu_machine_compute_time,
        machine_compute_time_size, cudaMemcpyDeviceToHost) != cudaSuccess) {

        fprintf(stderr, "[ERROR] Copiando machine_compute_time desde el dispositivo (%d bytes).\n", machine_compute_time_size);
        exit(EXIT_FAILURE);
    }
}

void comparar_sol_cpu_vs_gpu(struct matrix *etc_matrix, struct solution *current_solution, struct pals_gpu_rtask_instance &instance) {
    // Validación de la memoria del dispositivo.
    fprintf(stdout, ">> VALIDANDO MEMORIA GPU\n");

    int aux_task_assignment[etc_matrix->tasks_count];

    if (cudaMemcpy(aux_task_assignment, instance.gpu_task_assignment, etc_matrix->tasks_count * sizeof(int),
        cudaMemcpyDeviceToHost) != cudaSuccess) {

        fprintf(stderr, "[ERROR] Copiando task_assignment al host (%ld bytes).\n", etc_matrix->tasks_count * sizeof(int));
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < etc_matrix->tasks_count; i++) {
        if (current_solution->task_assignment[i] != aux_task_assignment[i]) {
            fprintf(stdout, "[INFO] task assignment diff => task %d on host: %d, on device: %d\n",
                i, current_solution->task_assignment[i], aux_task_assignment[i]);
        }
    }

    float aux_machine_compute_time[etc_matrix->machines_count];

    if (cudaMemcpy(aux_machine_compute_time, instance.gpu_machine_compute_time, etc_matrix->machines_count * sizeof(float),
        cudaMemcpyDeviceToHost) != cudaSuccess) {

        fprintf(stderr, "[ERROR] Copiando machine_compute_time al host (%ld bytes).\n", etc_matrix->machines_count * sizeof(float));
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < etc_matrix->machines_count; i++) {
        if (current_solution->machine_compute_time[i] != aux_machine_compute_time[i]) {
            fprintf(stdout, "[INFO] machine CT diff => machine %d on host: %f, on device: %f\n",
                i, current_solution->machine_compute_time[i], aux_machine_compute_time[i]);
        }
    }
}

void pals_gpu_rtask(struct params &input, struct matrix *etc_matrix, struct solution *current_solution) {
    // ==============================================================================
    // PALS aleatorio por tarea.
    // ==============================================================================

    // Time stop condition -----------------------------------------
    //timespec ts_stop_condition_start, ts_stop_condition_current;
    //clock_gettime(CLOCK_REALTIME, &ts_stop_condition_start);

    // Timming -----------------------------------------------------
    timespec ts_init;
    timming_start(ts_init);
    // Timming -----------------------------------------------------

    struct pals_gpu_rtask_instance instance;

    // Inicializo la memoria en el dispositivo.
    pals_gpu_rtask_init(etc_matrix, current_solution, instance);

    if (DEBUG) {
        comparar_sol_cpu_vs_gpu(etc_matrix, current_solution, instance);
    }

    // Timming -----------------------------------------------------
    timming_end(">> pals_gpu_rtask_init", ts_init);
    // Timming -----------------------------------------------------

    // ===========> DEBUG
    if (DEBUG) {
        validate_solution(etc_matrix, current_solution);
    }
    // <=========== DEBUG

    float makespan_inicial = current_solution->makespan;

    // Ejecuto GPUPALS.
    int seed = input.seed;

    //RNG_rand48 r48;
    //RNG_rand48_init(r48, PALS_RTASK_RANDS); // Debe ser múltiplo de 6144

    int prng_vector_size = PALS_RTASK_RANDS;
    //unsigned int prng_seeds[4] = {3822712292, 495793398, 4202624243, 3503457871}; // generated with: od -vAn -N4 -tu4 < /dev/urandom

    mtgp32_status mt_status;
    mtgp32_initialize(&mt_status, prng_vector_size, seed);

    // Cantidad de números aleatorios por invocación.
    unsigned int rand_iter_size = instance.blocks * 2;

    int prng_cant_iter_generadas = PALS_RTASK_RANDS / rand_iter_size;
    int prng_iter_actual = prng_cant_iter_generadas;

    if (DEBUG) fprintf(stdout, "[INFO] Cantidad de iteraciones por generación de numeros aleatorios: %d.\n", prng_cant_iter_generadas);

    int convergence_flag;
    convergence_flag = 0;

    int best_solution_iter = -1;
    float best_solution = VERY_BIG_FLOAT;

    //int makespan_idx_aux[COMPUTE_MAKESPAN_KERNEL_BLOCKS];
    float makespan_ct_aux[COMPUTE_MAKESPAN_KERNEL_BLOCKS];

    //clock_gettime(CLOCK_REALTIME, &ts_stop_condition_current);

    int iter;
    /*for (iter = 0; (iter < PALS_COUNT) && (convergence_flag == 0)
        && (ts_stop_condition_current.tv_sec - ts_stop_condition_start.tv_sec) <= 5; iter++) {*/
    for (iter = 0; (iter < PALS_COUNT) && (convergence_flag == 0); iter++) {

        if (DEBUG) fprintf(stdout, "[INFO] Iteracion %d =====================\n", iter);

        // ==============================================================================
        // Sorteo de numeros aleatorios.
        // ==============================================================================

        timespec ts_rand;
        timming_start(ts_rand);

        prng_iter_actual = prng_iter_actual + (instance.blocks * 2);

        if (prng_iter_actual >= prng_cant_iter_generadas) {
            if (DEBUG) fprintf(stdout, "[INFO] Generando %d números aleatorios...\n", PALS_RTASK_RANDS);

            //RNG_rand48_generate(r48, seed);
            mtgp32_generate_uint32(&mt_status);
            prng_iter_actual = 0;
        }

        timming_end(">> RNG_rand48", ts_rand);

        // Timming -----------------------------------------------------
        timespec ts_wrapper;
        timming_start(ts_wrapper);
        // Timming -----------------------------------------------------

        if (DEBUG) cudaThreadSynchronize();
        fprintf(stdout, "-1\n");
        if (cudaMemcpy(makespan_ct_aux, instance.gpu_makespan_ct_aux, sizeof(float) * COMPUTE_MAKESPAN_KERNEL_BLOCKS,
            cudaMemcpyDeviceToHost) != cudaSuccess) {

            fprintf(stderr, "[ERROR] Copiando gpu_makespan_ct_aux al host (%ld bytes).\n",
                COMPUTE_MAKESPAN_KERNEL_BLOCKS * sizeof(float));
            exit(EXIT_FAILURE);
        }

        fprintf(stdout, "0\n");
        if (cudaMemcpy(makespan_ct_aux, instance.gpu_makespan_ct_aux, sizeof(float) * COMPUTE_MAKESPAN_KERNEL_BLOCKS,
            cudaMemcpyDeviceToHost) != cudaSuccess) {

            fprintf(stderr, "[ERROR] Copiando gpu_makespan_ct_aux al host (%ld bytes).\n",
                COMPUTE_MAKESPAN_KERNEL_BLOCKS * sizeof(float));
            exit(EXIT_FAILURE);
        }

        /*pals_gpu_rtask_wrapper(etc_matrix, current_solution, instance,
            (int*)(&(mt_status.d_data[prng_iter_actual])));*/
        pals_gpu_rtask_wrapper(etc_matrix, current_solution, instance,
            (int*)mt_status.d_data);

        // Timming -----------------------------------------------------
        timming_end(">> pals_gpu_rtask_wrapper", ts_wrapper);
        // Timming -----------------------------------------------------

        // Timming -----------------------------------------------------
        timespec ts_post;
        timming_start(ts_post);
        // Timming -----------------------------------------------------

        if (DEBUG) cudaThreadSynchronize();
        fprintf(stdout, "1\n");
        if (cudaMemcpy(makespan_ct_aux, instance.gpu_makespan_ct_aux, sizeof(float) * COMPUTE_MAKESPAN_KERNEL_BLOCKS,
            cudaMemcpyDeviceToHost) != cudaSuccess) {

            fprintf(stderr, "[ERROR] Copiando gpu_makespan_ct_aux al host (%ld bytes).\n",
                COMPUTE_MAKESPAN_KERNEL_BLOCKS * sizeof(float));
            exit(EXIT_FAILURE);
        }

        pals_compute_makespan<<< COMPUTE_MAKESPAN_KERNEL_BLOCKS, COMPUTE_MAKESPAN_KERNEL_THREADS >>>(
            etc_matrix->machines_count, instance.gpu_machine_compute_time,
            instance.gpu_makespan_idx_aux, instance.gpu_makespan_ct_aux);

        /*
        if (cudaMemcpy(makespan_idx_aux, instance.gpu_makespan_idx_aux, sizeof(int) * COMPUTE_MAKESPAN_KERNEL_BLOCKS,
            cudaMemcpyDeviceToHost) != cudaSuccess) {

            fprintf(stderr, "[ERROR] Copiando gpu_makespan_idx_aux al host (%ld bytes).\n",
                COMPUTE_MAKESPAN_KERNEL_BLOCKS * sizeof(int));
            exit(EXIT_FAILURE);
        }
        * */

        if (DEBUG) cudaThreadSynchronize();
        fprintf(stdout, "2\n");
        if (cudaMemcpy(makespan_ct_aux, instance.gpu_makespan_ct_aux, sizeof(float) * COMPUTE_MAKESPAN_KERNEL_BLOCKS,
            cudaMemcpyDeviceToHost) != cudaSuccess) {

            fprintf(stderr, "[ERROR] Copiando gpu_makespan_ct_aux al host (%ld bytes).\n",
                COMPUTE_MAKESPAN_KERNEL_BLOCKS * sizeof(float));
            exit(EXIT_FAILURE);
        }

        if (DEBUG) {
            for (int i = 0; i < COMPUTE_MAKESPAN_KERNEL_BLOCKS; i++) {
                fprintf(stdout, "COMPUTE_MAKESPAN %f\n", makespan_ct_aux[i]);
            }
        }

        float old;
        old = current_solution->makespan;

        current_solution->makespan = makespan_ct_aux[0];

        for (int i = 1; i < COMPUTE_MAKESPAN_KERNEL_BLOCKS; i++) {
            if (current_solution->makespan < makespan_ct_aux[i]) {
                current_solution->makespan = makespan_ct_aux[i];
            }
        }

        if (current_solution->makespan < best_solution) {
            best_solution = current_solution->makespan;
            best_solution_iter = iter;
        }

        //if (old < current_solution->makespan) {
            //fprintf(stderr, "PUTA! on iteration %d\n", iter);
            fprintf(stderr, ">> makespan old %f\n", old);
            fprintf(stderr, ">> makespan new %f\n\n", current_solution->makespan);
        //}
        // Timming -----------------------------------------------------
        timming_end(">> pals_gpu_rtask_post", ts_post);
        // Timming -----------------------------------------------------

        load_sol_from_gpu(etc_matrix, instance, current_solution);
        if (DEBUG) validate_solution(etc_matrix, current_solution);
        if (DEBUG) refresh_solution(etc_matrix, current_solution);
        if (DEBUG) comparar_sol_cpu_vs_gpu(etc_matrix, current_solution, instance);

        //clock_gettime(CLOCK_REALTIME, &ts_stop_condition_current);
    }

    // Timming -----------------------------------------------------
    timespec ts_finalize;
    timming_start(ts_finalize);
    // Timming -----------------------------------------------------

    if (DEBUG) {
        fprintf(stdout, "[DEBUG] Total iterations       : %d.\n", iter);
        fprintf(stdout, "[DEBUG] Iter. best sol. found  : %d.\n", best_solution_iter);
        fprintf(stdout, "[DEBUG] Best sol. found        : %f.\n", best_solution);
        fprintf(stdout, "[DEBUG] Current sol.           : %f.\n", current_solution->makespan);

        fprintf(stdout, "[DEBUG] Current blocks count   : %d.\n", instance.blocks);
        fprintf(stdout, "[DEBUG] Current loops count    : %d.\n", instance.loops);
    }

    load_sol_from_gpu(etc_matrix, instance, current_solution);
    if (DEBUG) validate_solution(etc_matrix, current_solution);
    if (DEBUG) refresh_solution(etc_matrix, current_solution);
    if (DEBUG) comparar_sol_cpu_vs_gpu(etc_matrix, current_solution, instance);

    // Libera la memoria del dispositivo con los números aleatorios.
    //RNG_rand48_cleanup(r48);
    mtgp32_free(&mt_status);

    if (DEBUG) {
        fprintf(stdout, "[DEBUG] Viejo makespan: %f\n", makespan_inicial);
        fprintf(stdout, "[DEBUG] Nuevo makespan: %f\n", current_solution->makespan);
    } else {
        if (!OUTPUT_SOLUTION) fprintf(stdout, "%f\n", current_solution->makespan);

        fprintf(stderr, "CANT_ITERACIONES|%d\n", iter);
        fprintf(stderr, "BEST_FOUND|%d\n", best_solution_iter);
        /*fprintf(stderr, "TOTAL_SWAPS|%ld\n", cantidad_swaps);
        fprintf(stderr, "TOTAL_MOVES|%ld\n", cantidad_movs);*/
        fprintf(stderr, "TOTAL_SWAPS|0\n");
        fprintf(stderr, "TOTAL_MOVES|0\n");
        fprintf(stderr, "BEST_FOUND_VALUE|%f\n", best_solution);
        fprintf(stderr, "CURRENT_FOUND_VALUE|%f\n", current_solution->makespan);
    }

    // Libero la memoria del dispositivo.
    pals_gpu_rtask_finalize(instance);

    // Timming -----------------------------------------------------
    timming_end(">> pals_gpu_rtask_finalize", ts_finalize);
    // Timming -----------------------------------------------------
}
