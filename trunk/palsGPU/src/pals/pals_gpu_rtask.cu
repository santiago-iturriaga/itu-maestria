#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <limits.h>
#include <assert.h>

#include "../config.h"
#include "../utils.h"

#include "../random/cpu_rand.h"
#include "../random/RNG_rand48.h"

#include "pals_gpu_rtask.h"

#define VERY_BIG_FLOAT                  1073741824
#define PALS_RTASK_RANDS                6144*20

#define TOTALLY_FUCKUP_AUX_SIZE         6

#define PALS_GPU_RTASK__BLOCKS          128
#define PALS_GPU_RTASK__THREADS         256
#define PALS_GPU_RTASK__LOOPS           4

#define APPLY_BEST_KERNEL_BLOCKS        1
#define APPLY_BEST_KERNEL_THREADS       PALS_GPU_RTASK__BLOCKS >> 1

// No puedo trabajar con más de COMPUTE_MAKESPAN_KERNEL_THREADS * COMPUTE_MAKESPAN_KERNEL_BLOCKS machines.
// 512 * 2 = 1024
#define COMPUTE_MAKESPAN_KERNEL_BLOCKS        2
#define COMPUTE_MAKESPAN_KERNEL_THREADS       512

__global__ void pals_rtask_kernel(ushort machines_count, 
    ushort tasks_count, float current_makespan, float *gpu_etc_matrix, 
    ushort *gpu_task_assignment, float *gpu_machine_compute_time,
    int *gpu_random_numbers, int *gpu_best_movements_op,
    int *gpu_best_movements_thread, int *gpu_best_movements_loop,
    float *gpu_best_deltas)
{
    const unsigned int thread_idx = threadIdx.x;
    const unsigned int block_idx = blockIdx.x;
    const unsigned int block_dim = blockDim.x; // Cantidad de threads.

    const short mov_type = (short)(block_idx & 0x1);

    const unsigned int random1 = gpu_random_numbers[2 * block_idx];
    const unsigned int random2 = gpu_random_numbers[(2 * block_idx) + 1];

    __shared__ short block_operations[PALS_GPU_RTASK__THREADS];
    __shared__ ushort block_threads[PALS_GPU_RTASK__THREADS];
    __shared__ ushort block_loops[PALS_GPU_RTASK__THREADS];
    __shared__ float block_deltas[PALS_GPU_RTASK__THREADS];

    for (int loop = 0; loop < PALS_GPU_RTASK__LOOPS; loop++) {
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
        gpu_best_movements_op[block_idx] = (int)block_operations[0];    // Best movement operation.
        gpu_best_movements_thread[block_idx] = (int)block_threads[0];   // Best movement thread index.
        gpu_best_movements_loop[block_idx] = (int)block_loops[0];       // Best movement loop index.
        gpu_best_deltas[block_idx] = block_deltas[0];                   // Best movement delta.
    }
}

__global__ void pals_apply_best_kernel(ushort machines_count, 
    ushort tasks_count, float *gpu_etc_matrix, 
    ushort *gpu_task_assignment, float *gpu_machine_compute_time,
    int *gpu_random_numbers, int *gpu_best_movements_op, 
    int *gpu_best_movements_thread, int *gpu_best_movements_loop, 
    float *gpu_best_deltas, int *gpu_totally_fuckup_aux) {
    
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
        int thread_idx = gpu_best_movements_thread[i];
        int loop_idx = gpu_best_movements_loop[i];
        int random1 = gpu_random_numbers[2 * i];
        int random2 = gpu_random_numbers[(2 * i) + 1];
        
        gpu_totally_fuckup_aux[0] = mov_type;
        gpu_totally_fuckup_aux[1] = i;
        gpu_totally_fuckup_aux[2] = thread_idx;
        gpu_totally_fuckup_aux[3] = loop_idx;
        gpu_totally_fuckup_aux[4] = random1;
        gpu_totally_fuckup_aux[5] = random2;

        if (mov_type == 0) { // Comparación a nivel de bit para saber si es par o impar.
            // Si es impar...
            // Movimiento SWAP.

            ushort task_x, task_y;
            ushort machine_a, machine_b;

            // ================= Obtengo las tareas sorteadas.
            task_x = (random1 + loop_idx) % tasks_count;
            task_y = ((random2 >> 1) + (loop_idx * PALS_GPU_RTASK__THREADS)  + thread_idx) % (tasks_count - 1);
            if (task_y >= task_x) task_y++;

            // ================= Obtengo las máquinas a las que estan asignadas las tareas.
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
            // Si es par...
            // Movimiento MOVE.

            ushort task_x;
            ushort machine_a, machine_b;

            // ================= Obtengo la tarea sorteada, la máquina a la que esta asignada,
            // ================= y el compute time de la máquina.
            task_x = (random1 + loop_idx) % tasks_count;
            machine_a = gpu_task_assignment[task_x]; // Máquina a.

            // ================= Obtengo la máquina destino sorteada.
            machine_b = ((random2 >> 1) + (loop_idx * PALS_GPU_RTASK__THREADS) + thread_idx) % (machines_count - 1);
            if (machine_b >= machine_a) machine_b++;

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

__global__ void pals_compute_makespan(ushort machines_count, 
    float *gpu_machine_compute_time, int *machine_idx, float *machine_ct) {

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
    instance.blocks = PALS_GPU_RTASK__BLOCKS; //512; //32; //128; // NUNCA MÁS DE 512!!!
    instance.threads = PALS_GPU_RTASK__THREADS;
    instance.loops = PALS_GPU_RTASK__LOOPS; // 4; //32;

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
    if (cudaMalloc((void**)&(instance.gpu_best_movements_thread), best_movements_size) != cudaSuccess) {
        fprintf(stderr, "[ERROR] Solicitando memoria gpu_best_movements_thread (%d bytes).\n", best_movements_size);
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void**)&(instance.gpu_best_movements_loop), best_movements_size) != cudaSuccess) {
        fprintf(stderr, "[ERROR] Solicitando memoria gpu_best_movements_loop (%d bytes).\n", best_movements_size);
        exit(EXIT_FAILURE);
    }

    int best_deltas_size = sizeof(float) * instance.blocks;
    if (cudaMalloc((void**)&(instance.gpu_best_deltas), best_deltas_size) != cudaSuccess) {
        fprintf(stderr, "[ERROR] Solicitando memoria gpu_best_deltas (%d bytes).\n", best_deltas_size);
        exit(EXIT_FAILURE);
    }

    if (cudaMalloc((void**)&(instance.gpu_totally_fuckup_aux), sizeof(int) * TOTALLY_FUCKUP_AUX_SIZE) != cudaSuccess) {
        fprintf(stderr, "[ERROR] Solicitando memoria gpu_totally_fuckup_aux (%lu bytes).\n", sizeof(int) * TOTALLY_FUCKUP_AUX_SIZE);
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
    
    instance.cpu_totally_fuckup_aux = (int*)malloc(sizeof(int) * TOTALLY_FUCKUP_AUX_SIZE);

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
    int task_assignment_size = sizeof(short) * etc_matrix->tasks_count;
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

    if (cudaFree(instance.gpu_best_movements_thread) != cudaSuccess) {
        fprintf(stderr, "[ERROR] Liberando la memoria solicitada para gpu_best_movements_thread.\n");
        exit(EXIT_FAILURE);
    }

    if (cudaFree(instance.gpu_best_movements_loop) != cudaSuccess) {
        fprintf(stderr, "[ERROR] Liberando la memoria solicitada para gpu_best_movements_loop.\n");
        exit(EXIT_FAILURE);
    }
    
    if (cudaFree(instance.gpu_totally_fuckup_aux) != cudaSuccess) {
        fprintf(stderr, "[ERROR] Liberando la memoria solicitada para gpu_totally_fuckup_aux.\n");
        exit(EXIT_FAILURE);
    }
    
    free(instance.cpu_totally_fuckup_aux);    

    if (cudaFree(instance.gpu_makespan_idx_aux) != cudaSuccess) {
        fprintf(stderr, "[ERROR] Liberando la memoria solicitada para gpu_makespan_idx_aux.\n");
        exit(EXIT_FAILURE);
    }
    
    if (cudaFree(instance.gpu_makespan_ct_aux) != cudaSuccess) {
        fprintf(stderr, "[ERROR] Liberando la memoria solicitada para gpu_makespan_ct_aux.\n");
        exit(EXIT_FAILURE);
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

    cudaThreadSynchronize();

    dim3 grid(instance.blocks, 1, 1);
    dim3 threads(instance.threads, 1, 1);

    pals_rtask_kernel<<< grid, threads >>>(
        etc_matrix->machines_count,
        etc_matrix->tasks_count,
        s->makespan,
        instance.gpu_etc_matrix,
        instance.gpu_task_assignment,
        instance.gpu_machine_compute_time,
        gpu_random_numbers,
        instance.gpu_best_movements_op,
        instance.gpu_best_movements_thread,
        instance.gpu_best_movements_loop,
        instance.gpu_best_deltas);

    cudaError_t e;
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "[ERROR] Failure in kernel call.\n%s\n", cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }

    cudaThreadSynchronize();

    // Timming -----------------------------------------------------
    timming_end(".. pals_gpu_rtask_pals", ts_pals);
    // Timming -----------------------------------------------------

    // =====================================================================
    // Hago un reduce y aplico el mejor movimiento.
    // =====================================================================

    // Timming -----------------------------------------------------
    timespec ts_pals_reduce;
    timming_start(ts_pals_reduce);
    // Timming -----------------------------------------------------

    pals_apply_best_kernel<<< APPLY_BEST_KERNEL_BLOCKS, APPLY_BEST_KERNEL_THREADS >>>(
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

    //cudaThreadSynchronize();
    
    if (cudaMemcpy(instance.cpu_totally_fuckup_aux, instance.gpu_totally_fuckup_aux, sizeof(int) * TOTALLY_FUCKUP_AUX_SIZE, 
        cudaMemcpyDeviceToHost) != cudaSuccess) {
        
        fprintf(stderr, "[ERROR] Copiando gpu_totally_fuckup_aux.\n");
        exit(EXIT_FAILURE);            
    }

    int move_type = instance.cpu_totally_fuckup_aux[0];
    //int block_idx = instance.cpu_totally_fuckup_aux[1];
    int thread_idx = instance.cpu_totally_fuckup_aux[2];
    int loop_idx = instance.cpu_totally_fuckup_aux[3];
    int random1 = instance.cpu_totally_fuckup_aux[4];
    int random2 = instance.cpu_totally_fuckup_aux[5];
    
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
    
    if (move_type == PALS_GPU_RTASK_SWAP) { // Movement type: SWAP
        ushort task_x = (ushort)((random1 + loop_idx) % etc_matrix->tasks_count);
        ushort task_y = (ushort)(((random2 >> 1) + (loop_idx * instance.threads) + thread_idx) % (etc_matrix->tasks_count - 1));
        if (task_y >= task_x) task_y++;
       
        ushort machine_a = s->task_assignment[task_x];
        ushort machine_b = s->task_assignment[task_y];
        
        // Actualizo la asignación de cada tarea en el host.
        s->task_assignment[task_x] = machine_b;
        s->task_assignment[task_y] = machine_a;

        // Actualizo los compute time de cada máquina luego del move en el host.
        s->machine_compute_time[machine_a] = 
            s->machine_compute_time[machine_a] +
            get_etc_value(etc_matrix, machine_a, task_y) - 
            get_etc_value(etc_matrix, machine_a, task_x);

        s->machine_compute_time[machine_b] = 
            s->machine_compute_time[machine_b] +
            get_etc_value(etc_matrix, machine_b, task_x) - 
            get_etc_value(etc_matrix, machine_b, task_y);
       
        // =======> DEBUG
        if (DEBUG) { 
            ushort machine_a = s->task_assignment[task_x];
            ushort machine_b = s->task_assignment[task_y];

            fprintf(stdout, "[DEBUG] Task %d in %d swaps with task %d in %d.\n",
                task_x, machine_a, task_y, machine_b);
        }
        // <======= DEBUG
    } else if (move_type == PALS_GPU_RTASK_MOVE) { // Movement type: MOVE
        ushort task_x = (ushort)((random1 + loop_idx) % etc_matrix->tasks_count);
        ushort machine_a = s->task_assignment[task_x];
        ushort machine_b = (ushort)(((random2 >> 1) + (loop_idx * instance.threads) + thread_idx) % (etc_matrix->machines_count - 1));
        if (machine_b >= machine_a) machine_b++;

        s->task_assignment[task_x] = machine_b;
    
        // Actualizo los compute time de cada máquina luego del move en el host.
        s->machine_compute_time[machine_a] = 
            s->machine_compute_time[machine_a] - 
            get_etc_value(etc_matrix, machine_a, task_x);

        s->machine_compute_time[machine_b] = 
            s->machine_compute_time[machine_b] +
            get_etc_value(etc_matrix, machine_b, task_x);
   
        // =======> DEBUG
        if (DEBUG) {
            fprintf(stdout, "[DEBUG] Task %d in %d is moved to machine %d.\n",
                task_x, machine_a, machine_b);
        }
        // <======= DEBUG
    }

    // Timming -----------------------------------------------------
    timming_end(".. pals_gpu_rtask_pals_post", ts_pals_post);
    // Timming -----------------------------------------------------
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
        // Validación de la memoria del dispositivo.
        fprintf(stdout, ">> VALIDANDO MEMORIA GPU\n");

        ushort aux_task_assignment[etc_matrix->tasks_count];

        if (cudaMemcpy(aux_task_assignment, instance.gpu_task_assignment, etc_matrix->tasks_count * sizeof(short),
            cudaMemcpyDeviceToHost) != cudaSuccess) {

            fprintf(stderr, "[ERROR] Copiando task_assignment al host (%ld bytes).\n", etc_matrix->tasks_count * sizeof(short));
            exit(EXIT_FAILURE);
        }

        for (ushort i = 0; i < etc_matrix->tasks_count; i++) {
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

        for (ushort i = 0; i < etc_matrix->machines_count; i++) {
            if (current_solution->machine_compute_time[i] != aux_machine_compute_time[i]) {
                fprintf(stdout, "[INFO] machine CT diff => machine %d on host: %f, on device: %f\n",
                    i, current_solution->machine_compute_time[i], aux_machine_compute_time[i]);
            }
        }
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

    RNG_rand48 r48;
    RNG_rand48_init(r48, PALS_RTASK_RANDS); // Debe ser múltiplo de 6144

    // Cantidad de números aleatorios por invocación.
    unsigned int rand_iter_size = instance.blocks * 2;

    const short cant_iter_generadas = PALS_RTASK_RANDS / rand_iter_size;
    if (DEBUG) fprintf(stdout, "[INFO] Cantidad de iteraciones por generación de numeros aleatorios: %d.\n", cant_iter_generadas);

    ulong cantidad_swaps = 0;
    ulong cantidad_movs = 0;

    short convergence_flag;
    convergence_flag = 0;

    int best_solution_iter = -1;

    int makespan_idx_aux[COMPUTE_MAKESPAN_KERNEL_BLOCKS];
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

        if (iter % cant_iter_generadas == 0) {
            if (DEBUG) fprintf(stdout, "[INFO] Generando %d números aleatorios...\n", PALS_RTASK_RANDS);
            RNG_rand48_generate(r48, seed);
        }

        timming_end(">> RNG_rand48", ts_rand);

        // Timming -----------------------------------------------------
        timespec ts_wrapper;
        timming_start(ts_wrapper);
        // Timming -----------------------------------------------------

        pals_gpu_rtask_wrapper(etc_matrix, current_solution, instance,
            &(r48.res[(iter % cant_iter_generadas) * rand_iter_size]));

        // Timming -----------------------------------------------------
        timming_end(">> pals_gpu_rtask_wrapper", ts_wrapper);
        // Timming -----------------------------------------------------

        // Timming -----------------------------------------------------
        timespec ts_post;
        timming_start(ts_post);
        // Timming -----------------------------------------------------

        pals_compute_makespan<<< COMPUTE_MAKESPAN_KERNEL_BLOCKS, COMPUTE_MAKESPAN_KERNEL_THREADS >>>(
            etc_matrix->machines_count, instance.gpu_machine_compute_time,
            instance.gpu_makespan_idx_aux, instance.gpu_makespan_ct_aux);

        if (cudaMemcpy(makespan_idx_aux, instance.gpu_makespan_idx_aux, sizeof(int) * COMPUTE_MAKESPAN_KERNEL_BLOCKS,
            cudaMemcpyDeviceToHost) != cudaSuccess) {

            fprintf(stderr, "[ERROR] Copiando gpu_makespan_idx_aux al host (%ld bytes).\n", 
                COMPUTE_MAKESPAN_KERNEL_BLOCKS * sizeof(int));
            exit(EXIT_FAILURE);
        }

        if (cudaMemcpy(makespan_ct_aux, instance.gpu_makespan_ct_aux, sizeof(float) * COMPUTE_MAKESPAN_KERNEL_BLOCKS,
            cudaMemcpyDeviceToHost) != cudaSuccess) {

            fprintf(stderr, "[ERROR] Copiando gpu_makespan_ct_aux al host (%ld bytes).\n", 
                COMPUTE_MAKESPAN_KERNEL_BLOCKS * sizeof(float));
            exit(EXIT_FAILURE);
        }

        current_solution->makespan = makespan_ct_aux[0];

        for (ushort i = 1; i < COMPUTE_MAKESPAN_KERNEL_BLOCKS; i++) {
            if (current_solution->makespan < makespan_ct_aux[i]) {
                current_solution->makespan = makespan_ct_aux[i];
            }
        }        

        if (DEBUG) {
            fprintf(stdout, "[DEBUG] makespan: %f...", current_solution->makespan);
            
            float verificar_makespan;
            verificar_makespan = current_solution->machine_compute_time[0];

            for (ushort i = 1; i < etc_matrix->machines_count; i++) {
                if (verificar_makespan < current_solution->machine_compute_time[i]) {
                    verificar_makespan = current_solution->machine_compute_time[i];
                }
            }
            
            assert(verificar_makespan == current_solution->makespan);
            fprintf(stdout, " OK\n");
        }

        if (DEBUG) {
            // Validación de la memoria del dispositivo.
            fprintf(stdout, ">> VALIDANDO MEMORIA GPU\n");

            ushort aux_task_assignment[etc_matrix->tasks_count];

            if (cudaMemcpy(aux_task_assignment, instance.gpu_task_assignment, etc_matrix->tasks_count * sizeof(short),
                cudaMemcpyDeviceToHost) != cudaSuccess) {

                fprintf(stderr, "[ERROR] Copiando task_assignment al host (%ld bytes).\n", etc_matrix->tasks_count * sizeof(short));
                exit(EXIT_FAILURE);
            }

            for (ushort i = 0; i < etc_matrix->tasks_count; i++) {
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

            for (ushort i = 0; i < etc_matrix->machines_count; i++) {
                if (current_solution->machine_compute_time[i] != aux_machine_compute_time[i]) {
                    fprintf(stdout, "[INFO] machine CT diff => machine %d on host: %f, on device: %f\n",
                        i, current_solution->machine_compute_time[i], aux_machine_compute_time[i]);
                }
            }
        }

        /*
        if ((cantidad_movs_iter > 0) || (cantidad_swaps_iter > 0)) {
            // Actualiza el makespan de la solución.
            // Si cambio el makespan, busco el nuevo makespan.
            ushort machine = 0;
            current_solution->makespan = current_solution->machine_compute_time[0];

            for (ushort i = 1; i < etc_matrix->machines_count; i++) {
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

        if (increase_depth >= 5) {
            //if (increase_depth >= 500) {
            //if ((iter-best_solution_iter) >= (etc_matrix->machines_count * 1000)) {
                
            convergence_flag = 1;
            //increase_depth = 0;
        }
        * */

        // Timming -----------------------------------------------------
        timming_end(">> pals_gpu_rtask_post", ts_post);
        // Timming -----------------------------------------------------

        // Nuevo seed.
        seed++;

        //clock_gettime(CLOCK_REALTIME, &ts_stop_condition_current);
    }

    // Timming -----------------------------------------------------
    timespec ts_finalize;
    timming_start(ts_finalize);
    // Timming -----------------------------------------------------

    if (DEBUG) {
        fprintf(stdout, "[DEBUG] Total iterations       : %d.\n", iter);
        fprintf(stdout, "[DEBUG] Iter. best sol. found  : %d.\n", best_solution_iter);

        fprintf(stdout, "[DEBUG] Total swaps performed  : %ld.\n", cantidad_swaps);
        fprintf(stdout, "[DEBUG] Total movs performed   : %ld.\n", cantidad_movs);

        fprintf(stdout, "[DEBUG] Current blocks count   : %d.\n", instance.blocks);
        fprintf(stdout, "[DEBUG] Current loops count    : %d.\n", instance.loops);
    }

    if (DEBUG) {
        // Validación de la memoria del dispositivo.
        fprintf(stdout, ">> VALIDANDO MEMORIA GPU\n");

        ushort aux_task_assignment[etc_matrix->tasks_count];

        if (cudaMemcpy(aux_task_assignment, instance.gpu_task_assignment, etc_matrix->tasks_count * sizeof(short),
            cudaMemcpyDeviceToHost) != cudaSuccess) {

            fprintf(stderr, "[ERROR] Copiando task_assignment al host (%ld bytes).\n", etc_matrix->tasks_count * sizeof(short));
            exit(EXIT_FAILURE);
        }

        for (ushort i = 0; i < etc_matrix->tasks_count; i++) {
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

        for (ushort i = 0; i < etc_matrix->machines_count; i++) {
            if (current_solution->machine_compute_time[i] != aux_machine_compute_time[i]) {
                fprintf(stdout, "[INFO] machine CT diff => machine %d on host: %f, on device: %f\n",
                    i, current_solution->machine_compute_time[i], aux_machine_compute_time[i]);
            }
        }
    }

    // Libera la memoria del dispositivo con los números aleatorios.
    RNG_rand48_cleanup(r48);

    // Reconstruye el compute time de cada máquina.
    // NOTA: tengo que hacer esto cada tanto por errores acumulados en el redondeo.
    for (ushort i = 0; i < etc_matrix->machines_count; i++) {
        current_solution->machine_compute_time[i] = 0.0;
    }

    for (ushort i = 0; i < etc_matrix->tasks_count; i++) {
        ushort assigned_machine = current_solution->task_assignment[i];

        current_solution->machine_compute_time[assigned_machine] =
            current_solution->machine_compute_time[assigned_machine] +
            get_etc_value(etc_matrix, assigned_machine, i);
    }

    // Actualiza el makespan de la solución.
    current_solution->makespan = current_solution->machine_compute_time[0];
    for (ushort i = 1; i < etc_matrix->machines_count; i++) {
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
    pals_gpu_rtask_finalize(instance);

    // Timming -----------------------------------------------------
    timming_end(">> pals_gpu_rtask_finalize", ts_finalize);
    // Timming -----------------------------------------------------
}
