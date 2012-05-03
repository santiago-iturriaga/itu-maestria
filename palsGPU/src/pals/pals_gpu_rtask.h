/*
 * pals.h
 *
 *  Created on: Jul 27, 2011
 *      Author: santiago
 */

#include "../etc_matrix.h"
#include "../solution.h"

#define PALS_GPU_RTASK_SWAP 0
#define PALS_GPU_RTASK_MOVE 1

#ifndef PALS_GPU_RTASK_H_
#define PALS_GPU_RTASK_H_

struct pals_gpu_rtask_instance {
    // Datos del estado actual del problema a resolver.
    float *gpu_etc_matrix;
    int *gpu_task_assignment;
    float *gpu_machine_compute_time;
    
    // Espacio de memoria en el dispositivo para almacenar los
    // mejores movimientos encontrados en cada iteración.
    int *gpu_best_movements_op;
    int *gpu_best_movements_data1;
    int *gpu_best_movements_data2;
    float *gpu_best_deltas;
    int *gpu_best_movements_discarded;
    
    int *gpu_makespan_idx_aux;
    float *gpu_makespan_ct_aux;
    
    int *cpu_best_movements_op;
    int *cpu_best_movements_data1;
    int *cpu_best_movements_data2;
    float *cpu_best_deltas;
    
    char *result_task_history;
    char *result_machine_history;
    
    // Parámetros de ejecución del kernel.
    ushort blocks;
    ushort threads;
    ushort loops;
    
    // Cantidad de movimientos probados por iteración.
    ulong total_tasks;
};

/*
 * Ejecuta el algoritmo.
 * Búsqueda masivamente paralela sobre un subdominio del problema. 
 * Se sortea el subdominio por tarea.
 */
void pals_gpu_rtask(struct params &input, struct matrix *etc_matrix, 
    struct solution *current_solution);

/*
 * Reserva e inicializa la memoria del dispositivo con los datos del problema.
 */
void pals_gpu_rtask_init(struct matrix *etc_matrix, struct solution *s, 
    struct pals_gpu_rtask_instance &instance);

/*
 * Libera la memoria del dispositivo.
 */
void pals_gpu_rtask_finalize(struct pals_gpu_rtask_instance &instance);

/*
 * Ejecuta PALS en el dispositivo.
 */
void pals_gpu_rtask_wrapper(struct matrix *etc_matrix, struct solution *s, 
    struct pals_gpu_rtask_instance &instance, int *gpu_random_numbers);

#endif /* PALS_GPU_H_ */
