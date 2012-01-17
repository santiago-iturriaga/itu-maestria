#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <assert.h>
#include <string.h>
#include <pthread.h>

#include "../config.h"
#include "../utils.h"
#include "../random/cpu_rand.h"

#include "pals_cpu_rtask.h"

#define RANDOM_NUMBERS_PER_THREAD_ITER 5

void pals_cpu_rtask(struct params &input, struct matrix *etc_matrix, struct solution *current_solution) {	
	// ==============================================================================
	// PALS aleatorio por tarea.
	// ==============================================================================
	
	// Timming -----------------------------------------------------
	timespec ts_init;
	timming_start(ts_init);
	// Timming -----------------------------------------------------

	float makespan_inicial = get_makespan(current_solution);

	// Inicializo la memoria y los hilos de ejecución.
	struct pals_cpu_rtask_instance instance;
	pals_cpu_rtask_init(input, etc_matrix, current_solution, input.seed, instance);
    
	// Timming -----------------------------------------------------
	timming_end(">> pals_cpu_rtask_init", ts_init);
	// Timming -----------------------------------------------------

    // Bloqueo la ejecución hasta que terminen todos los hilos.
    if(pthread_join(*(instance.master_thread), NULL))
    {
        printf("Could not join master thread\n");
        exit(EXIT_FAILURE);
    }
    
    for(int i = 0; i < instance.count_threads; i++)
    {
        if(pthread_join(instance.slave_threads[i], NULL))
        {
            printf("Could not join slave thread %d\n", i);
            exit(EXIT_FAILURE);
        }
    }
	
	// Timming -----------------------------------------------------
	timespec ts_finalize;
	timming_start(ts_finalize);
	// Timming -----------------------------------------------------

    // TODO: Hacer algo con las soluciones!!!
		
	// ===========> DEBUG
	if (DEBUG) {
		fprintf(stdout, "[INFO] Cantidad de iteraciones        : %d\n", instance.total_iterations);
		fprintf(stdout, "[INFO] Último mejor encontrado en     : %d\n", instance.last_elite_found_on_iter);
		fprintf(stdout, "[INFO] Total de makespan searches     : %d\n", instance.total_makespan_greedy_searches);
		fprintf(stdout, "[INFO] Total de energy searches       : %d\n", instance.total_energy_greedy_searches);
		fprintf(stdout, "[INFO] Total de random searches       : %d\n", instance.total_random_greedy_searches);
		fprintf(stdout, "[INFO] Total de swaps                 : %ld\n", instance.total_swaps);
		fprintf(stdout, "[INFO] Total de moves                 : %ld\n", instance.total_moves);
	
		validate_solution(current_solution);
	}
	// <=========== DEBUG
	
	if (DEBUG) {
		fprintf(stdout, "[DEBUG] Viejo makespan: %f\n", makespan_inicial);
		fprintf(stdout, "[DEBUG] Nuevo makespan: %f\n", get_makespan(current_solution));
		fprintf(stdout, "[DEBUG] Mejora: %.2f\n", 100 - (get_makespan(current_solution) / makespan_inicial) * 100);
	} else {
	        if (!OUTPUT_SOLUTION) fprintf(stdout, "%f\n", get_makespan(current_solution));
        	fprintf(stderr, "CANT_ITERACIONES|%d\n", instance.total_iterations);
        	fprintf(stderr, "BEST_FOUND|%d\n", instance.last_elite_found_on_iter);
	        fprintf(stderr, "TOTAL_SWAPS|%ld\n", instance.total_swaps);
        	fprintf(stderr, "TOTAL_MOVES|%ld\n", instance.total_moves);
	        fprintf(stderr, "TOTAL_RANDOM_SEARCHES|%d\n", instance.total_makespan_greedy_searches);
	        fprintf(stderr, "TOTAL_ENERGY_SEARCHES|%d\n", instance.total_energy_greedy_searches);
	        fprintf(stderr, "TOTAL_MAKESPAN_SEARCHES|%d\n", instance.total_random_greedy_searches);
	}

	// Libero la memoria del dispositivo.
	pals_cpu_rtask_finalize(instance);
	
	// Timming -----------------------------------------------------
	timming_end(">> pals_cpu_rtask_finalize", ts_finalize);
	// Timming -----------------------------------------------------		
}

void pals_cpu_rtask_init(struct params &input, struct matrix *etc_matrix, struct solution *s, int seed,
	struct pals_cpu_rtask_instance &empty_instance) {
	
	// Asignación del paralelismo del algoritmo.
	empty_instance.count_threads = input.thread_count;
	
	// Cantidad de resultados retornados por iteración.
	empty_instance.result_count = empty_instance.count_threads - 1;
	
	if (DEBUG) {
		fprintf(stdout, "[INFO] Seed                                    : %d\n", seed);
		fprintf(stdout, "[INFO] Number of threads                       : %d\n", empty_instance.count_threads);
		fprintf(stdout, "[INFO] PALS_CPU_RTASK_WORK__CONVERGENCE        : %d\n", PALS_CPU_RTASK_WORK__CONVERGENCE);
		fprintf(stdout, "[INFO] PALS_CPU_RTASK_WORK__ELITE_POP_MAX_SIZE : %d\n", PALS_CPU_RTASK_WORK__ELITE_POP_MAX_SIZE);
        fprintf(stdout, "[INFO] PALS_CPU_RTASK_WORK__SRC_TASK_NHOOD     : %d\n", PALS_CPU_RTASK_WORK__SRC_TASK_NHOOD);
        fprintf(stdout, "[INFO] PALS_CPU_RTASK_WORK__DST_TASK_NHOOD     : %d\n", PALS_CPU_RTASK_WORK__DST_TASK_NHOOD);
	}

    // =========================================================================
    // Pido la memoria e inicializo la solución de partida.
    
    empty_instance.etc_matrix = etc_matrix;   
    empty_instance.initial_solution = s;

    empty_instance.elite_population = (struct solution*)malloc(sizeof(struct solution) * PALS_CPU_RTASK_WORK__ELITE_POP_MAX_SIZE);
    if (empty_instance.elite_population == NULL) {
		fprintf(stderr, "[ERROR] Solicitando memoria para elite population.\n");
		exit(EXIT_FAILURE);
	}
        
    empty_instance.elite_population_status = (int*)malloc(sizeof(int) * PALS_CPU_RTASK_WORK__ELITE_POP_MAX_SIZE);
    if (empty_instance.elite_population_status == NULL) {
		fprintf(stderr, "[ERROR] Solicitando memoria para elite population status.\n");
		exit(EXIT_FAILURE);
	}
    
    for (int i = 0; i < PALS_CPU_RTASK_WORK__ELITE_POP_MAX_SIZE; i++) {
        empty_instance.elite_population_status = PALS_CPU_RTASK_WORK__ELITE_POP_EMPTY;
    }

    empty_instance.elite_population_count = 1;
    clone_solution(&(empty_instance.elite_population[0]), s);

	empty_instance.__result_task_history = (char*)malloc(sizeof(char) * etc_matrix->tasks_count);
	empty_instance.__result_machine_history = (char*)malloc(sizeof(char) * etc_matrix->machines_count);

	// =========================================================================
	// Pedido de memoria para la generación de numeros aleatorios.
	
	timespec ts_1;
	timming_start(ts_1);
	
	srand(seed);
	long int random_seed;
	
	empty_instance.random_states = (struct cpu_rand_state*)malloc(sizeof(struct cpu_rand_state) * empty_instance.count_threads);
	
	for (int i = 0; i < empty_instance.count_threads; i++) {
	    random_seed = rand();
	    cpu_rand_init(random_seed, empty_instance.random_states[i]);
	}
	
	empty_instance.slave_random_numbers = (double*)malloc(sizeof(double) * (empty_instance.count_threads - 1) * RANDOM_NUMBERS_PER_THREAD_ITER);
	
	timming_end(".. cpu_rand_buffers", ts_1);
	
	// =========================================================================
	// Pedido de memoria para almacenar los mejores movimientos de cada iteración.
	
	empty_instance.move_type = (int*)malloc(sizeof(int) * empty_instance.result_count);
	empty_instance.origin = (int*)malloc(sizeof(int) * empty_instance.result_count);
	empty_instance.destination = (int*)malloc(sizeof(int) * empty_instance.result_count);
	empty_instance.delta_makespan = (float*)malloc(sizeof(float) * empty_instance.result_count);
	empty_instance.delta_ct = (float*)malloc(sizeof(float) * empty_instance.result_count);
	empty_instance.delta_energy = (float*)malloc(sizeof(float) * empty_instance.result_count);

	// =========================================================================
	// Creo e inicializo los threads y los mecanismos de sincronización del sistema.
	
	timespec ts_threads;
	timming_start(ts_threads);
	
	empty_instance.sync_barrier = (pthread_barrier_t*)malloc(sizeof(pthread_barrier_t));
	if (pthread_barrier_init(empty_instance.sync_barrier, NULL, empty_instance.count_threads))
    {
        printf("Could not create a barrier\n");
        exit(EXIT_FAILURE);
    }

	empty_instance.master_thread = (pthread_t*)malloc(sizeof(pthread_t));
	
	empty_instance.slave_threads = (pthread_t*)
	    malloc(sizeof(pthread_t) * (empty_instance.count_threads - 1));
	
	empty_instance.slave_threads_args = (struct pals_cpu_rtask_thread_arg*)
	    malloc(sizeof(struct pals_cpu_rtask_thread_arg) * (empty_instance.count_threads - 1));

    empty_instance.slave_work_type = (int*)malloc(sizeof(int) * (empty_instance.count_threads - 1));
    empty_instance.slave_work_solution_idx = (int*)malloc(sizeof(int) * (empty_instance.count_threads - 1));

    // Creo el hilo master.
    if (pthread_create(empty_instance.master_thread, NULL, pals_cpu_rtask_master_thread, (void*) &(empty_instance)))
    {
        printf("Could not create master thread\n");
        exit(EXIT_FAILURE);
    }
	
	// Creo los hilos esclavos.
	for (int i = 0; i < (empty_instance.count_threads - 1); i++) {
   		empty_instance.slave_threads_args[i].thread_idx = i;
   		
        empty_instance.slave_threads_args[i].etc_matrix = etc_matrix;
        
        empty_instance.slave_threads_args[i].population = empty_instance.elite_population;
        empty_instance.slave_threads_args[i].population_status = empty_instance.elite_population_status;
        empty_instance.slave_threads_args[i].population_count = &(empty_instance.elite_population_count);
        
        empty_instance.slave_threads_args[i].work_solution_idx = &(empty_instance.slave_work_solution_idx[i]);
    	empty_instance.slave_threads_args[i].work_type = &(empty_instance.slave_work_type[i]);
        empty_instance.slave_threads_args[i].sync_barrier = empty_instance.sync_barrier;
        	
        empty_instance.slave_threads_args[i].thread_random_state = &(empty_instance.random_states[i]);
        empty_instance.slave_threads_args[i].thread_random_numbers = &(empty_instance.slave_random_numbers[i * RANDOM_NUMBERS_PER_THREAD_ITER]);
        
        empty_instance.slave_threads_args[i].thread_move_type = &(empty_instance.move_type[i]);
        empty_instance.slave_threads_args[i].thread_origin = &(empty_instance.origin[i]);
        empty_instance.slave_threads_args[i].thread_destination = &(empty_instance.destination[i]);
        empty_instance.slave_threads_args[i].thread_delta_makespan = &(empty_instance.delta_makespan[i]);
        empty_instance.slave_threads_args[i].thread_delta_ct = &(empty_instance.delta_ct[i]);
        empty_instance.slave_threads_args[i].thread_delta_energy = &(empty_instance.delta_energy[i]);
   		
        if (pthread_create(&(empty_instance.slave_threads[i]), NULL, pals_cpu_rtask_slave_thread, 
            (void*) &(empty_instance.slave_threads_args[i])))
        {
            printf("Could not create slave thread %d\n", i);
            exit(EXIT_FAILURE);
        }
	}

	timming_end(".. thread creation", ts_threads);
}

void pals_cpu_rtask_finalize(struct pals_cpu_rtask_instance &instance) {      
    free(instance.elite_population);
    free(instance.elite_population_status);
    
    free(instance.random_states);
    free(instance.slave_random_numbers);
        
	free(instance.move_type);
	free(instance.origin);
	free(instance.destination);
	free(instance.delta_makespan);
	free(instance.delta_ct);
	free(instance.delta_energy);

	free(instance.slave_threads);	
	free(instance.slave_threads_args);
	free(instance.slave_work_type);
    free(instance.slave_work_solution_idx);
		
	free(instance.sync_barrier);
	
	free(instance.__result_task_history);
	free(instance.__result_machine_history);
}

void* pals_cpu_rtask_master_thread(void *thread_arg) {

    struct pals_cpu_rtask_instance *instance;
    instance = (struct pals_cpu_rtask_instance *)thread_arg;

	// ===========> DEBUG
	if (DEBUG) {
	    //TODO: validar todas?
		//validate_solution(instance->current_solution);
	}
	// <=========== DEBUG

    int rc;
	
	instance->total_iterations = 0;
    instance->last_elite_found_on_iter = -1;
    instance->total_makespan_greedy_searches = 0;
    instance->total_energy_greedy_searches = 0;
    instance->total_random_greedy_searches = 0;
	instance->total_swaps = 0;
    instance->total_moves = 0;
	
	int increase_depth = 0;
	int convergence_flag = 0;

    // Genero numeros aletorios para la primer iteración.    
    cpu_rand_generate(instance->random_states[instance->count_threads-1], PALS_CPU_RTASK_MASTER_RANDOMS, instance->master_random_numbers);
    
	int iter;
	for (iter = 0; (iter < PALS_COUNT) && (convergence_flag == 0); iter++) {
		if (DEBUG) fprintf(stdout, "[INFO] Iteracion %d =====================\n", iter);

		// Timming -----------------------------------------------------
		timespec ts_search;
		timming_start(ts_search);
		// Timming -----------------------------------------------------

        // ===========================================================================
        // Asigno el trabajo para los hilos.
        int current_work_type;
        current_work_type = (int)floor(instance->master_random_numbers[0] * 3);

        int current_work_solution_idx;
        current_work_solution_idx = (int)floor(instance->master_random_numbers[1] * instance->elite_population_count);
        
        for (int i = 0; i < instance->count_threads - 1; i++) {
            instance->slave_work_type[i] = current_work_type;
            instance->slave_work_solution_idx[i] = current_work_solution_idx;

            current_work_type = (current_work_type + 1) % 3;       
            current_work_solution_idx = (current_work_solution_idx + 1) % instance->elite_population_count;
        }

        // ===========================================================================
        // Mando a los esclavos a buscar movimientos.
        rc = pthread_barrier_wait(instance->sync_barrier);
        if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
        {
            printf("Could not wait on barrier\n");
            exit(EXIT_FAILURE);
        }

        // ===========================================================================
        // Mientras los hilos buscan, genero numeros aleatorios
        cpu_rand_generate(instance->random_states[instance->count_threads-1], PALS_CPU_RTASK_MASTER_RANDOMS, instance->master_random_numbers);
        
        // ===========================================================================
        // Espero a que los esclavos terminen.
        rc = pthread_barrier_wait(instance->sync_barrier);
        if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
        {
            printf("Could not wait on barrier\n");
            exit(EXIT_FAILURE);
        }        
                
		// Timming -----------------------------------------------------
		timming_end(">> pals_cpu_rtask_search", ts_search);
		// Timming -----------------------------------------------------
        
		// Timming -----------------------------------------------------
		timespec ts_post;
		timming_start(ts_post);
		// Timming -----------------------------------------------------

		// Aplico los movimientos.
		memset(instance->__result_task_history, 0, instance->etc_matrix->tasks_count);
		memset(instance->__result_machine_history, 0, instance->etc_matrix->machines_count);
		
		int cantidad_swaps_iter, cantidad_moves_iter;
		cantidad_swaps_iter = 0;
		cantidad_moves_iter = 0;
	
	    // Busco el thread que encontró el mejor movimiento.
	    int initial_idx = (int)floor(instance->master_random_numbers[2] * instance->result_count);
		
	    for (int i = 0; i < instance->result_count; i++) {
		    int result_idx = (initial_idx + i) % instance->result_count;

			//if (DEBUG) fprintf(stdout, "[DEBUG] Movement %d, delta = %f.\n", result_idx, instance->delta[result_idx]);

            // Nunca hago movimiento negativos sobre los objetivos.
			if ((instance->delta_makespan[result_idx] <= 0.0) && (instance->delta_energy[result_idx] <= 0.0)) {	
			    if (instance->move_type[result_idx] == PALS_CPU_RTASK_SWAP) {
			        /*
				    int task_x = instance->origin[result_idx];
				    int task_y = instance->destination[result_idx];
		
				    int machine_a = get_task_assignment(instance->current_solution, instance->origin[result_idx]);
				    int machine_b = get_task_assignment(instance->current_solution, instance->destination[result_idx]);
		
				    if (DEBUG) fprintf(stdout, "        (swap) Task %d in %d swaps with task %d in %d. Delta %f.\n",
					    instance->origin[result_idx], machine_a, instance->destination[result_idx], machine_b, instance->delta[result_idx]);
		
				    if ((instance->__result_task_history[task_x] == 0) && (instance->__result_task_history[task_y] == 0) &&
					    (instance->__result_machine_history[machine_a] == 0) && (instance->__result_machine_history[machine_b] == 0))	{

					    cantidad_swaps_iter++;
		
					    instance->__result_task_history[task_x] = 1;
					    instance->__result_task_history[task_y] = 1;
					    instance->__result_machine_history[machine_a] = 1;
					    instance->__result_machine_history[machine_b] = 1;
					
					    if (DEBUG) {
						    fprintf(stdout, ">> [pre-update]:\n");
						    fprintf(stdout, "   machine_a: %d, old_machine_a_ct: %f.\n", machine_a, 
						        get_machine_compute_time(instance->current_solution, machine_a));
						    fprintf(stdout, "   machine_b: %d, old_machine_b_ct: %f.\n", machine_b, 
						        get_machine_compute_time(instance->current_solution, machine_b));
					    }
		
		                swap_tasks(instance->current_solution, task_x, task_y);

					    if (DEBUG) {
						    fprintf(stdout, ">> [update]:\n");
						    fprintf(stdout, "   task_x: %d, task_x_machine: %d.\n", task_x, machine_b);
						    fprintf(stdout, "   task_y: %d, task_y_machine: %d.\n", task_y, machine_a);
						    fprintf(stdout, "   machine_a: %d, machine_a_ct: %f.\n", machine_a, 
						        get_machine_compute_time(instance->current_solution, machine_a));
						    fprintf(stdout, "   machine_b: %d, machine_b_ct: %f.\n", machine_b, 
						        get_machine_compute_time(instance->current_solution, machine_b));
						    fprintf(stdout, "   old_makespan: %f.\n", get_makespan(instance->current_solution));
					    }
				    } else {
					    if (DEBUG) fprintf(stdout, "[DEBUG] Lo ignoro porque una tarea o máquina de este movimiento ya fue modificada.\n");
				    }*/
			    } else if (instance->move_type[result_idx] == PALS_CPU_RTASK_MOVE) {
				    /*int task_x = instance->origin[result_idx];
				    int machine_a = get_task_assignment(instance->current_solution, task_x);
				    int machine_b = instance->destination[result_idx];

				    if (DEBUG) fprintf(stdout, "        (move) Task %d in %d is moved to machine %d. Delta %f.\n",
					    instance->origin[result_idx], machine_a, instance->destination[result_idx], instance->delta[result_idx]);
				
				    if ((instance->__result_task_history[task_x] == 0) &&
     					(instance->__result_machine_history[machine_a] == 0) &&
					    (instance->__result_machine_history[machine_b] == 0))	{
		
					    cantidad_movs_iter++;
		
					    instance->__result_task_history[task_x] = 1;
					    instance->__result_machine_history[machine_a] = 1;
					    instance->__result_machine_history[machine_b] = 1;
				
			            if (DEBUG) {
						    fprintf(stdout, ">> [pre-update]:\n");
						    fprintf(stdout, "   machine_a: %d, old_machine_a_ct: %f.\n", machine_a, 
						        get_machine_compute_time(instance->current_solution, machine_a));
						    fprintf(stdout, "   machine_b: %d, old_machine_b_ct: %f.\n", machine_b, 
						        get_machine_compute_time(instance->current_solution, machine_b));
					    }
			
        				move_task_to_machine(instance->current_solution, task_x, machine_b);
			
					    if (DEBUG) {
						    fprintf(stdout, ">> [update]:\n");
						    fprintf(stdout, "   task_x: %d, task_x_machine: %d.\n", task_x, machine_b);
						    fprintf(stdout, "   machine_a: %d, machine_a_ct: %f.\n", machine_a, 
						        get_machine_compute_time(instance->current_solution, machine_a));
						    fprintf(stdout, "   machine_b: %d, machine_b_ct: %f.\n", machine_b, 
						        get_machine_compute_time(instance->current_solution, machine_b));
						    fprintf(stdout, "   old_makespan: %f.\n", get_makespan(instance->current_solution));
					    }
				    } else {
					    if (DEBUG) fprintf(stdout, "[DEBUG] Lo ignoro porque una tarea o máquina de este movimiento ya fue modificada.\n");
				    }*/
			    }
			}
		}

		if ((cantidad_moves_iter > 0) || (cantidad_swaps_iter > 0)) {
			if (DEBUG) {
				fprintf(stdout, "   swaps performed  : %d.\n", cantidad_swaps_iter);
				fprintf(stdout, "   movs performed   : %d.\n", cantidad_moves_iter);
			}
			
			instance->total_swaps += cantidad_swaps_iter;
			instance->total_moves += cantidad_moves_iter;
		}
		
		if (instance->last_elite_found_on_iter == iter) {
			increase_depth = 0;
		
			if (DEBUG) {
				fprintf(stdout, "   elite solution found.\n");
			}
		} else {
			increase_depth++;

			if (DEBUG) {
				fprintf(stdout, "   elite population unchanged (%d).\n", increase_depth);
			}
		}

		if (increase_depth >= PALS_CPU_RTASK_WORK__CONVERGENCE) {
			convergence_flag = 1;
		}

		// Timming -----------------------------------------------------
		timming_end(">> pals_cpu_rtask_post", ts_post);
		// Timming -----------------------------------------------------
	}
	
    // Mando a los esclavos a terminar.
    for (int i = 0; i < instance->count_threads - 1; i++) {
        instance->slave_work_type[i] = PALS_CPU_RTASK_WORK__DO_EXIT;
    }

    rc = pthread_barrier_wait(instance->sync_barrier);
    if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
    {
        printf("Could not wait on barrier\n");
        exit(EXIT_FAILURE);
    }
        	
	return NULL;
}

void* pals_cpu_rtask_slave_thread(void *thread_arg)
{	
    int rc;
    pals_cpu_rtask_thread_arg *thread_instance;

    thread_instance = (pals_cpu_rtask_thread_arg*)thread_arg;

    // Genero los números aleatorios necesarios para esta iteración (por las dudas).
    cpu_rand_generate(*(thread_instance->thread_random_state), RANDOM_NUMBERS_PER_THREAD_ITER, thread_instance->thread_random_numbers);
    
    // Espero a que el master asigne trabajo.
    rc = pthread_barrier_wait(thread_instance->sync_barrier);
    if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
    {
        printf("Could not wait on barrier\n");
        exit(EXIT_FAILURE);
    }

    while (*(thread_instance->work_type) != PALS_CPU_RTASK_WORK__DO_EXIT) {
        // Busco la solución con la que me toca trabajar.
        int current_solution_idx = -1;
        int current_solution_pos;
        
        for (current_solution_pos = 0; (current_solution_pos < PALS_CPU_RTASK_WORK__ELITE_POP_MAX_SIZE) 
            && (current_solution_idx < *(thread_instance->work_solution_idx)); current_solution_pos++) {
            
            if (thread_instance->population_status[current_solution_pos] == PALS_CPU_RTASK_WORK__ELITE_POP_SOL) {
                current_solution_idx++;
            }
        }

        // Determino que movimientos se van a realizar.
       	int mov_type;
       	if (thread_instance->thread_random_numbers[0] < 0.75) {
       	    mov_type = PALS_CPU_RTASK_SWAP;
       	} else {
       	    mov_type = PALS_CPU_RTASK_MOVE;
       	}

	    double random_machine_a = thread_instance->thread_random_numbers[1];
	    double random_machine_b = thread_instance->thread_random_numbers[2];

	    int machine_a, machine_b;
        machine_a = (int)floor(random_machine_a * thread_instance->etc_matrix->machines_count);
        machine_b = (int)floor(random_machine_b * thread_instance->etc_matrix->machines_count);

        if (machine_a == machine_b) machine_b++;

	    double random_task_x = thread_instance->thread_random_numbers[3];
	    double random_task_y = thread_instance->thread_random_numbers[4];
	    
	    int task_x_id, task_y_id;
   	    int task_x_offset, task_y_offset;
	    int task_x_pos = -1, task_y_pos = -1;
	    int task_x_aux, task_y_aux;
	    
	    int tasks_in_machine_a = get_machine_tasks_count(&(thread_instance->population[current_solution_pos]), machine_a);
	    int tasks_in_machine_b = get_machine_tasks_count(&(thread_instance->population[current_solution_pos]), machine_b);
	    
        task_x_aux = (int)floor(random_task_x * tasks_in_machine_a);
        task_y_aux = (int)floor(random_task_y * tasks_in_machine_b);

        /*
        int *tasks_a = get_all_tasks(&(thread_instance->population[current_solution_pos]), machine_a);
        int *tasks_b = get_all_tasks(&(thread_instance->population[current_solution_pos]), machine_b);

        for (task_x_offset = 0; (tasks_a[task_x_offset] != MACHINE__EOT) && (task_x_pos != task_x_aux); task_x_offset++) {
            if (tasks_a[task_x_offset] != MACHINE__EMPTY) task_x_pos++;
        }
        
        task_x_id = tasks_a[task_x_offset];
        
        for (task_y_offset = 0; (tasks_b[task_y_offset] != MACHINE__EOT) && (task_y_pos != task_y_aux); task_y_offset++) {
            if (tasks_b[task_y_offset] != MACHINE__EMPTY) task_y_pos++;
        }

        task_y_id = tasks_b[task_y_offset];*/

   	    int task_y_offset_current;
        for (int src_nhood_idx = 0; src_nhood_idx < PALS_CPU_RTASK_WORK__SRC_TASK_NHOOD; src_nhood_idx++) {
            task_y_offset_current = task_y_offset;
        
            for (int dst_nhood_idx = 0; dst_nhood_idx < PALS_CPU_RTASK_WORK__DST_TASK_NHOOD; dst_nhood_idx++) {            
	            // Tipo de movimiento.
	            if (mov_type == PALS_CPU_RTASK_SWAP) {
		            // Movimiento SWAP.
	
	                /*
		            float machine_a_ct_old, machine_b_ct_old;
		            float machine_a_ct_new, machine_b_ct_new;
		            float machine_a_energy_old, machine_b_energy_old;
		            float machine_a_energy_new, machine_b_energy_new;
	
		            float delta_makespan;
		            delta_makespan = 0.0;

		            float delta_ct;
		            delta_ct = 0.0;
		            
                    float delta_energy;
		            delta_energy = 0.0;


	                // ================= Obtengo las tareas sorteadas.
                    

            	    for (int eval = 0; eval < thread_instance->count_evals; eval++) {	
		                // ================= Obtengo las tareas sorteadas (2).
			
		                task_y = (int)(floor((random2 * (thread_instance->etc_matrix->tasks_count - 1)) + (loop * thread_instance->count_evals) + eval)) 
		                    % (thread_instance->etc_matrix->tasks_count - 1);
		                if (task_y >= task_x) task_y++;
	
		                // ================= Obtengo las máquinas a las que estan asignadas las tareas.
		                machine_a = get_task_assignment(thread_instance->current_solution, task_x); // Máquina a.	
		                machine_b = get_task_assignment(thread_instance->current_solution, task_y); // Máquina b.	

		                if (machine_a != machine_b) {
			                // Calculo el delta del swap sorteado.
		
			                // Máquina 1.
			                machine_a_ct_old = get_machine_compute_time(thread_instance->current_solution, machine_a);
				
			                machine_a_ct_new = machine_a_ct_old;
			                machine_a_ct_new = machine_a_ct_new - get_etc_value(thread_instance->etc_matrix, machine_a, task_x); // Resto del ETC de x en a.
			                machine_a_ct_new = machine_a_ct_new + get_etc_value(thread_instance->etc_matrix, machine_a, task_y); // Sumo el ETC de y en a.
		
			                // Máquina 2.
			                machine_b_ct_old = get_machine_compute_time(thread_instance->current_solution, machine_b);

			                machine_b_ct_new = machine_b_ct_old;
			                machine_b_ct_new = machine_b_ct_new - get_etc_value(thread_instance->etc_matrix, machine_b, task_y); // Resto el ETC de y en b.
			                machine_b_ct_new = machine_b_ct_new + get_etc_value(thread_instance->etc_matrix, machine_a, task_x); // Sumo el ETC de x en b.

			                if ((machine_a_ct_new > get_makespan(thread_instance->current_solution)) 
			                    || (machine_b_ct_new > get_makespan(thread_instance->current_solution))) {
			                    
				                // Luego del movimiento aumenta el makespan. Intento desestimularlo lo más posible.
				                if (machine_a_ct_new > get_makespan(thread_instance->current_solution)) 
				                    delta = delta + (machine_a_ct_new - get_makespan(thread_instance->current_solution));
				                if (machine_b_ct_new > get_makespan(thread_instance->current_solution)) 
				                    delta = delta + (machine_b_ct_new - get_makespan(thread_instance->current_solution));
			                } else if ((machine_a_ct_old+1 >= get_makespan(thread_instance->current_solution)) 
			                    || (machine_b_ct_old+1 >= get_makespan(thread_instance->current_solution))) {	
			                    
				                // Antes del movimiento una las de máquinas definía el makespan. Estos son los mejores movimientos.
			
				                if (machine_a_ct_old+1 >= get_makespan(thread_instance->current_solution)) {
					                delta = delta + (machine_a_ct_new - machine_a_ct_old);
				                } else {
					                delta = delta + 1/(machine_a_ct_new - machine_a_ct_old);
				                }
			
				                if (machine_b_ct_old+1 >= get_makespan(thread_instance->current_solution)) {
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

		                if (((loop == 0) && (eval == 0))|| (thread_instance->thread_delta[0] > delta)) {			        
		                	thread_instance->thread_move_type[0] = PALS_CPU_RTASK_SWAP;
                            thread_instance->thread_origin[0] = task_x;
                            thread_instance->thread_destination[0] = task_y;
                            thread_instance->thread_delta[0] = delta;
		                }
		            }*/
	            } else {
		            // Movimiento MOVE.
	
	                /*
		            int task_x;
		            int machine_a, machine_b;
	
		            float machine_a_ct_old, machine_b_ct_old;
		            float machine_a_ct_new, machine_b_ct_new;

		            float delta;
		            delta = 0.0;

		            // ================= Obtengo la tarea sorteada.
            		task_x = (int)(floor(random1 * (thread_instance->etc_matrix->tasks_count - thread_instance->count_loops))) + loop;
	
            	    for (int eval = 0; eval < thread_instance->count_evals; eval++) {	
		                // ================= Obtengo la máquina a la que esta asignada,
		                // ================= y el compute time de la máquina.
		                machine_a = get_task_assignment(thread_instance->current_solution, task_x); // Máquina a.
		                machine_a_ct_old = get_machine_compute_time(thread_instance->current_solution, machine_a);
						
		                // ================= Obtengo la máquina destino sorteada.	
		                machine_b = (int)(floor((random2 * (thread_instance->etc_matrix->machines_count - 1)) + (loop * thread_instance->count_evals) + eval)) 
		                    % (thread_instance->etc_matrix->machines_count - 1);
		                if (machine_b >= machine_a) machine_b++;
	
		                machine_b_ct_old = get_machine_compute_time(thread_instance->current_solution, machine_b);
	
		                // Calculo el delta del swap sorteado.
		                machine_a_ct_new = machine_a_ct_old - get_etc_value(thread_instance->etc_matrix, machine_a, task_x); // Resto del ETC de x en a.		
		                machine_b_ct_new = machine_b_ct_old + get_etc_value(thread_instance->etc_matrix, machine_b, task_x); // Sumo el ETC de x en b.

		                if (machine_b_ct_new > get_makespan(thread_instance->current_solution)) {
			                // Luego del movimiento aumenta el makespan. Intento desestimularlo lo más posible.
			                delta = delta + (machine_b_ct_new - get_makespan(thread_instance->current_solution));
		                } else if (machine_a_ct_old+1 >= get_makespan(thread_instance->current_solution)) {	
			                // Antes del movimiento una las de máquinas definía el makespan. Estos son los mejores movimientos.
			                delta = delta + (machine_a_ct_new - machine_a_ct_old);
			                delta = delta + 1/(machine_b_ct_new - machine_b_ct_old);
		                } else {
			                // Ninguna de las máquinas intervenía en el makespan. Intento favorecer lo otros movimientos.
			                delta = delta + (machine_a_ct_new - machine_a_ct_old);
			                delta = delta + (machine_b_ct_new - machine_b_ct_old);
			                delta = 1 / delta;
		                }
		                
		                if (((loop == 0) && (eval == 0))|| (thread_instance->thread_delta[0] > delta)) {			        
		                	thread_instance->thread_move_type[0] = PALS_CPU_RTASK_MOVE;
                            thread_instance->thread_origin[0] = task_x;
                            thread_instance->thread_destination[0] = machine_b;
                            thread_instance->thread_delta[0] = delta;
		                }
                    }*/
	            }
	            
	            /*
	            if (dst_nhood_idx + 1 < PALS_CPU_RTASK_WORK__DST_TASK_NHOOD) {
                    for (task_y_offset_current++; (tasks_b[task_y_offset_current] == MACHINE__EOT) && (tasks_b[task_y_offset_current] == MACHINE__EMPTY); task_y_offset++) {
                        if (tasks_b[task_y_offset_current] == MACHINE__EOT) task_y_offset_current = 0;
                    }

                    task_y_id = tasks_b[task_y_offset_current];
                }*/
            }
            
            /*
            if (src_nhood_idx + 1 < PALS_CPU_RTASK_WORK__SRC_TASK_NHOOD) {
                for (task_x_offset++; (tasks_a[task_x_offset] == MACHINE__EOT) && (tasks_a[task_x_offset] == MACHINE__EMPTY); task_x_offset++) {
                    if (tasks_a[task_x_offset] == MACHINE__EOT) task_x_offset_current = 0;
                }
                
                task_x_id = tasks_a[task_x_offset];
            }*/
        }

        // Espero a que todos los slave threads terminen.
        rc = pthread_barrier_wait(thread_instance->sync_barrier);
        if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
        {
            printf("Could not wait on barrier\n");
            exit(EXIT_FAILURE);
        }

        // Genero los números aleatorios necesarios para esta iteración (por las dudas).
        cpu_rand_generate(*(thread_instance->thread_random_state), RANDOM_NUMBERS_PER_THREAD_ITER, thread_instance->thread_random_numbers);
    
        // Espero a que el master asigne trabajo.
        rc = pthread_barrier_wait(thread_instance->sync_barrier);
        if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
        {
            printf("Could not wait on barrier\n");
            exit(EXIT_FAILURE);
        }    
    } 

	return NULL;
}
