#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <assert.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <semaphore.h>

#include "../config.h"
#include "../utils.h"
#include "../basic/mct.h"
#include "../random/cpu_rand.h"
#include "../random/cpu_mt.h"

#include "pals_cpu_1pop.h"

void validate_thread_instance(struct pals_cpu_1pop_thread_arg *instance) {
    pthread_mutex_lock(instance->population_mutex);
    int cantidad = 0;
    
    for (int i = 0; i < instance->population_max_size; i++) {
        if (instance->population[i].status > 0) {
            cantidad++;
        }
    }
    
    if (cantidad != *(instance->population_count)) {
        fprintf(stdout, "[DEBUG] Population:\n");
	    fprintf(stdout, "        Expected population count: %d\n", *(instance->population_count));
	    fprintf(stdout, "        Real population count: %d\n", cantidad);
        for (int j = 0; j < instance->population_max_size; j++) {
            fprintf(stdout, "        [%d] status      %d\n", j, instance->population[j].status);
            fprintf(stdout, "        [%d] initialized %d\n", j, instance->population[j].initialized);
        }
    }

    pthread_mutex_unlock(instance->population_mutex);
    
    assert(cantidad == *(instance->population_count));
}

void pals_cpu_1pop(struct params &input, struct etc_matrix *etc, struct energy_matrix *energy) {
	// ==============================================================================
	// PALS aleatorio por tarea.
	// ==============================================================================
	
	// Timming -----------------------------------------------------
	timespec ts_init;
	timming_start(ts_init);
	// Timming -----------------------------------------------------

	// Inicializo la memoria y los hilos de ejecución.
	struct pals_cpu_1pop_instance instance;
	pals_cpu_1pop_init(input, etc, energy, input.seed, instance);
    
	// Timming -----------------------------------------------------
	timming_end(">> pals_cpu_1pop_init", ts_init);
	// Timming -----------------------------------------------------

    // Bloqueo la ejecución hasta que terminen todos los hilos.  
    for(int i = 0; i < instance.count_threads; i++)
    {
        if(pthread_join(instance.threads[i], NULL))
        {
            printf("Could not join thread %d\n", i);
            exit(EXIT_FAILURE);
        } else {
            if (DEBUG) printf("[DEBUG] thread %d <OK>\n", i);
        }
    }
	
	// Timming -----------------------------------------------------
	timespec ts_finalize;
	timming_start(ts_finalize);
	// Timming -----------------------------------------------------
	
	// ===========> DEBUG
	if (!OUTPUT_SOLUTION) {
	    int total_iterations = 0;
        int total_makespan_greedy_searches = 0;
        int total_energy_greedy_searches = 0;
        int total_random_greedy_searches = 0;
	    int total_swaps = 0;
        int total_moves = 0;
        int total_population_full = 0;

        for (int i = 0; i < instance.count_threads; i++) {
            total_iterations += instance.threads_args[i].total_iterations;
            total_makespan_greedy_searches += instance.threads_args[i].total_makespan_greedy_searches;
            total_energy_greedy_searches += instance.threads_args[i].total_energy_greedy_searches;
            total_random_greedy_searches += instance.threads_args[i].total_random_greedy_searches;
            total_swaps += instance.threads_args[i].total_swaps;
            total_moves += instance.threads_args[i].total_moves;
            total_population_full += instance.threads_args[i].total_population_full;
        }
	
		fprintf(stdout, "[INFO] Cantidad de iteraciones        : %d\n", total_iterations);
		fprintf(stdout, "[INFO] Total de makespan searches     : %d\n", total_makespan_greedy_searches);
		fprintf(stdout, "[INFO] Total de energy searches       : %d\n", total_energy_greedy_searches);
		fprintf(stdout, "[INFO] Total de random searches       : %d\n", total_random_greedy_searches);
		fprintf(stdout, "[INFO] Total de swaps                 : %d\n", total_swaps);
		fprintf(stdout, "[INFO] Total de moves                 : %d\n", total_moves);
		fprintf(stdout, "[INFO] Total poblacion llena          : %d\n", total_population_full);
		fprintf(stdout, "[INFO] Cantidad de soluciones         : %d\n", instance.population_count);

        if (DEBUG_DEV) {
	        for (int i = 0; i < instance.population_max_size; i++) {
				if (instance.population[i].status > SOLUTION__STATUS_EMPTY) {
					validate_solution(&(instance.population[i]));
				}
		    }
		}
	}
	// <=========== DEBUG

	if (DEBUG) {	
        fprintf(stdout, "== Population =================================================\n");
	    for (int i = 0; i < instance.population_max_size; i++) {
	        if (instance.population[i].status > SOLUTION__STATUS_EMPTY) {
                fprintf(stdout, "Solucion %d: %f %f\n", i, get_makespan(&(instance.population[i])), get_energy(&(instance.population[i])));
            }
        }
	} else {
	        if (!OUTPUT_SOLUTION) { 
	            fprintf(stdout, "== Population =================================================\n");
        	    for (int i = 0; i < instance.population_max_size; i++) {
        	        if (instance.population[i].status > SOLUTION__STATUS_EMPTY) {
    	                fprintf(stdout, "%f %f\n", get_makespan(&(instance.population[i])), get_energy(&(instance.population[i])));
    	            }
    	        }
	        } else {
				fprintf(stdout, "%d\n", instance.population_count);
        	    for (int i = 0; i < instance.population_max_size; i++) {
        	        if (instance.population[i].status > SOLUTION__STATUS_EMPTY) {
						for (int task = 0; task < etc->tasks_count; task++) {
							fprintf(stdout, "%d\n", get_task_assigned_machine_id(&(instance.population[i]), task));
						}
        	        }
        	    }
        	    
				// TODO:................
				/*fprintf(stderr, "CANT_ITERACIONES|%d\n", instance.total_iterations);
				fprintf(stderr, "BEST_FOUND|%d\n", instance.last_elite_found_on_iter);
				fprintf(stderr, "TOTAL_SWAPS|%ld\n", instance.total_swaps);
				fprintf(stderr, "TOTAL_MOVES|%ld\n", instance.total_moves);
				fprintf(stderr, "TOTAL_RANDOM_SEARCHES|%d\n", instance.total_makespan_greedy_searches);
				fprintf(stderr, "TOTAL_ENERGY_SEARCHES|%d\n", instance.total_energy_greedy_searches);
				fprintf(stderr, "TOTAL_MAKESPAN_SEARCHES|%d\n", instance.total_random_greedy_searches);*/
			}
	}

	// Libero la memoria del dispositivo.
	pals_cpu_1pop_finalize(instance);
	
	// Timming -----------------------------------------------------
	timming_end(">> pals_cpu_1pop_finalize", ts_finalize);
	// Timming -----------------------------------------------------		
}

void pals_cpu_1pop_init(struct params &input, struct etc_matrix *etc, struct energy_matrix *energy,
    int seed, struct pals_cpu_1pop_instance &empty_instance) {
    
	// Asignación del paralelismo del algoritmo.
	empty_instance.count_threads = input.thread_count;
	
	if (!OUTPUT_SOLUTION) {
		fprintf(stdout, "[INFO] == Input arguments ==================================\n");
		fprintf(stdout, "       Seed                                    : %d\n", seed);
		fprintf(stdout, "       Number of tasks                         : %d\n", etc->tasks_count);
		fprintf(stdout, "       Number of machines                      : %d\n", etc->machines_count);
		fprintf(stdout, "       Number of threads                       : %d\n", empty_instance.count_threads);
		fprintf(stdout, "[INFO] == Configuration constants ==========================\n");
		fprintf(stdout, "       PALS_CPU_1POP_WORK__TIMEOUT            : %d\n", PALS_CPU_1POP_WORK__TIMEOUT);
		fprintf(stdout, "       PALS_CPU_1POP_WORK__CONVERGENCE        : %d\n", PALS_CPU_1POP_WORK__CONVERGENCE);
		fprintf(stdout, "       PALS_CPU_1POP_SEARCH_OP_BALANCE__SWAP  : %f\n", PALS_CPU_1POP_SEARCH_OP_BALANCE__SWAP);
		fprintf(stdout, "       PALS_CPU_1POP_SEARCH_OP_BALANCE__MOVE  : %f\n", PALS_CPU_1POP_SEARCH_OP_BALANCE__MOVE);
		fprintf(stdout, "       PALS_CPU_1POP_SEARCH_BALANCE__MAKESPAN : %f\n", PALS_CPU_1POP_SEARCH_BALANCE__MAKESPAN);
		fprintf(stdout, "       PALS_CPU_1POP_SEARCH_BALANCE__ENERGY   : %f\n", PALS_CPU_1POP_SEARCH_BALANCE__ENERGY);
		fprintf(stdout, "       PALS_CPU_1POP_WORK__POP_SIZE_FACTOR    : %d (size=%d)\n", 
			PALS_CPU_1POP_WORK__POP_SIZE_FACTOR, PALS_CPU_1POP_WORK__POP_SIZE_FACTOR * empty_instance.count_threads);
		fprintf(stdout, "       PALS_CPU_1POP_WORK__THREAD_CONVERGENCE : %d\n", PALS_CPU_1POP_WORK__THREAD_CONVERGENCE);
		fprintf(stdout, "       PALS_CPU_1POP_WORK__THREAD_ITERATIONS  : %d\n", PALS_CPU_1POP_WORK__THREAD_ITERATIONS);
        fprintf(stdout, "       PALS_CPU_1POP_WORK__SRC_TASK_NHOOD     : %d\n", PALS_CPU_1POP_WORK__SRC_TASK_NHOOD);
        fprintf(stdout, "       PALS_CPU_1POP_WORK__DST_TASK_NHOOD     : %d\n", PALS_CPU_1POP_WORK__DST_TASK_NHOOD);
        fprintf(stdout, "       PALS_CPU_1POP_WORK__DST_MACH_NHOOD     : %d\n", PALS_CPU_1POP_WORK__DST_MACH_NHOOD);
		fprintf(stdout, "[INFO] =====================================================\n");
	}

    // =========================================================================
    // Pido la memoria e inicializo la solución de partida.

    empty_instance.etc = etc;
    empty_instance.energy = energy;
    
    empty_instance.population_max_size = PALS_CPU_1POP_WORK__POP_SIZE_FACTOR * empty_instance.count_threads;
	empty_instance.population_count = 0;

	// Population.
    empty_instance.population = (struct solution*)malloc(sizeof(struct solution) * empty_instance.population_max_size);
    if (empty_instance.population == NULL) {
		fprintf(stderr, "[ERROR] Solicitando memoria para population.\n");
		exit(EXIT_FAILURE);
	}
        
    for (int i = 0; i < empty_instance.population_max_size; i++) {
        empty_instance.population[i].status = SOLUTION__STATUS_EMPTY;
        empty_instance.population[i].initialized = 0;
    }
	
	// =========================================================================
	// Pedido de memoria para la generación de numeros aleatorios.
	
	timespec ts_1;
	timming_start(ts_1);
	
	srand(seed);
	long int random_seed;
	
	#ifdef CPU_MERSENNE_TWISTER
	empty_instance.random_states = (struct cpu_mt_state*)malloc(sizeof(struct cpu_mt_state) * empty_instance.count_threads);
	#else
	empty_instance.random_states = (struct cpu_rand_state*)malloc(sizeof(struct cpu_rand_state) * empty_instance.count_threads);
	#endif
	
	for (int i = 0; i < empty_instance.count_threads; i++) {
	    random_seed = rand();
	    
		#ifdef CPU_MERSENNE_TWISTER
	    cpu_mt_init(random_seed, empty_instance.random_states[i]);
	    #else
	    cpu_rand_init(random_seed, empty_instance.random_states[i]);
	    #endif
	}
	
	timming_end(".. cpu_rand_buffers", ts_1);
	
	// =========================================================================
	// Creo e inicializo los threads y los mecanismos de sincronización del sistema.
	
	timespec ts_threads;
	timming_start(ts_threads);

	if (pthread_mutex_init(&(empty_instance.work_type_mutex), NULL))
    {
        printf("Could not create a work type mutex\n");
        exit(EXIT_FAILURE);
    }

	if (pthread_mutex_init(&(empty_instance.population_mutex), NULL))
    {
        printf("Could not create a population mutex\n");
        exit(EXIT_FAILURE);
    }
	
	if (pthread_barrier_init(&(empty_instance.sync_barrier), NULL, empty_instance.count_threads))
    {
        printf("Could not create a sync barrier\n");
        exit(EXIT_FAILURE);
    }
	
	if (sem_init(&(empty_instance.new_solutions_sem), 0, 0))
    {
        printf("Could not create a new solutions sem\n");
        exit(EXIT_FAILURE);
    }

	// Creo los hilos.
	empty_instance.threads = (pthread_t*)
	    malloc(sizeof(pthread_t) * empty_instance.count_threads);
	
	empty_instance.threads_args = (struct pals_cpu_1pop_thread_arg*)
	    malloc(sizeof(struct pals_cpu_1pop_thread_arg) * empty_instance.count_threads);

	empty_instance.work_type = PALS_CPU_1POP_WORK__INIT;
	
	timespec ts_start;
	clock_gettime(CLOCK_REALTIME, &ts_start);
	
	for (int i = 0; i < empty_instance.count_threads; i++) {
   		empty_instance.threads_args[i].thread_idx = i;
   		
        empty_instance.threads_args[i].etc = empty_instance.etc;
        empty_instance.threads_args[i].energy = empty_instance.energy;
        
        empty_instance.threads_args[i].population = empty_instance.population;
        empty_instance.threads_args[i].population_count = &(empty_instance.population_count);
        empty_instance.threads_args[i].population_max_size = empty_instance.population_max_size;
        
    	empty_instance.threads_args[i].work_type = &(empty_instance.work_type);

        empty_instance.threads_args[i].work_type_mutex = &(empty_instance.work_type_mutex);		
        empty_instance.threads_args[i].population_mutex = &(empty_instance.population_mutex);
		empty_instance.threads_args[i].new_solutions_sem = &(empty_instance.new_solutions_sem);
		empty_instance.threads_args[i].sync_barrier = &(empty_instance.sync_barrier);
        	
        empty_instance.threads_args[i].thread_random_state = &(empty_instance.random_states[i]);
   		empty_instance.threads_args[i].ts_start = ts_start;
   		empty_instance.threads_args[i].total_population_full = 0;
   		
        if (pthread_create(&(empty_instance.threads[i]), NULL, pals_cpu_1pop_thread,  (void*) &(empty_instance.threads_args[i])))
        {
            printf("Could not create slave thread %d\n", i);
            exit(EXIT_FAILURE);
        }
	}

	timming_end(".. thread creation", ts_threads);
}

void pals_cpu_1pop_finalize(struct pals_cpu_1pop_instance &instance) {
	for (int i = 0; i < instance.population_max_size; i++) {
		if (instance.population[i].initialized == 1) {
			free_solution(&(instance.population[i]));
		}
	}
	    
    free(instance.population);
    free(instance.random_states);
	free(instance.threads);	
	free(instance.threads_args);

	pthread_mutex_destroy(&(instance.work_type_mutex));	
	pthread_mutex_destroy(&(instance.population_mutex));
	pthread_barrier_destroy(&(instance.sync_barrier));
	sem_destroy(&(instance.new_solutions_sem));
}

int pals_cpu_1pop_eval_new_solution(struct pals_cpu_1pop_thread_arg *instance, int new_solution_pos) {
	int solutions_deleted = 0;	
	int new_solution_is_dominated = 0;

	float makespan_new, energy_new;
	makespan_new = get_makespan(&(instance->population[new_solution_pos]));
	energy_new = get_energy(&(instance->population[new_solution_pos]));
	
	int s_idx = 0;
	for (int s_pos = 0; (s_pos < instance->population_max_size) && 
		(s_idx < *(instance->population_count)) && (new_solution_is_dominated == 0); s_pos++) {
		
		if ((instance->population[s_pos].status > SOLUTION__STATUS_EMPTY) && (s_pos != new_solution_pos)) {
			s_idx++;
			
			// Calculo no dominancia del elemento nuevo con el actual.
			float makespan, energy;
			makespan = get_makespan(&(instance->population[s_pos]));
			energy = get_energy(&(instance->population[s_pos]));

			if ((makespan <= makespan_new) && (energy <= energy_new)) {
				// La nueva solución es dominada por una ya existente.
				new_solution_is_dominated = 1;
			} else if ((makespan_new < makespan) && (energy_new < energy)) {
				// La nueva solución domina a una ya existente.
				solutions_deleted++;
				instance->population_count[0] = instance->population_count[0] - 1;
				instance->population[s_pos].status = SOLUTION__STATUS_EMPTY;
			} else {
				// Ninguna de las dos soluciones es dominada por la otra.
				
			}
		}
	}
	
	if (new_solution_is_dominated == 0) {
		if (*(instance->population_count) + instance->count_threads < instance->population_max_size) {
			// Retorno 1 porque la solución es no dominada.
			return 1;
		} else {
			//if (*(instance->population_count) + instance->count_threads == instance->population_max_size) {
			//}
			instance->total_population_full++;
			
			return 0;
		}
	} else {
		return 0;
	}
}

void* pals_cpu_1pop_thread(void *thread_arg) {	
    int rc;

    struct pals_cpu_1pop_thread_arg *thread_instance;
    thread_instance = (pals_cpu_1pop_thread_arg*)thread_arg;

	thread_instance->total_iterations = 0;
    thread_instance->total_makespan_greedy_searches = 0;
    thread_instance->total_energy_greedy_searches = 0;
    thread_instance->total_random_greedy_searches = 0;
	thread_instance->total_swaps = 0;
    thread_instance->total_moves = 0;
	   
	int terminate = 0; 
	int work_type = -1;

	timespec ts_current;
	clock_gettime(CLOCK_REALTIME, &ts_current);

    while ((terminate == 0) && (ts_current.tv_sec - thread_instance->ts_start.tv_sec < PALS_CPU_1POP_WORK__TIMEOUT)) {
	
        work_type = *(thread_instance->work_type);
        if (DEBUG_DEV) printf("[DEBUG] [THREAD=%d] Work type = %d\n", thread_instance->thread_idx, work_type);
    
        if (work_type == PALS_CPU_1POP_WORK__EXIT) {
			// PALS_CPU_1POP_WORK__EXIT =======================================================================
			// Finalizo la ejecución del algoritmo!
            terminate = 1;
            
        } else if (work_type == PALS_CPU_1POP_WORK__INIT) {
            // PALS_CPU_1POP_WORK__INIT_POP ===================================================================

	        // Timming -----------------------------------------------------
	        timespec ts_mct;
	        timming_start(ts_mct);
	        // Timming -----------------------------------------------------

			if (thread_instance->thread_idx < thread_instance->population_max_size) {
				// Inicializo el individuo que me tocó.
				init_empty_solution(thread_instance->etc, thread_instance->energy, &(thread_instance->population[thread_instance->thread_idx]));
				
				#ifdef CPU_MERSENNE_TWISTER
				double random = cpu_mt_generate(*(thread_instance->thread_random_state));
				#else
				double random = cpu_rand_generate(*(thread_instance->thread_random_state));
				#endif
				
				int random_task = (int)floor(random * thread_instance->etc->tasks_count);
				compute_custom_mct(&(thread_instance->population[thread_instance->thread_idx]), random_task);

				thread_instance->population[thread_instance->thread_idx].status = SOLUTION__STATUS_NEW;
				
				pthread_mutex_lock(thread_instance->work_type_mutex);
					thread_instance->population_count[0] = thread_instance->population_count[0] + 1;
				pthread_mutex_unlock(thread_instance->work_type_mutex);

				//if (DEBUG) {
					fprintf(stdout, "[DEBUG] MCTing individual %d (%f %f)\n", 
						thread_instance->thread_idx, get_makespan(&(thread_instance->population[thread_instance->thread_idx])), 
						get_energy(&(thread_instance->population[thread_instance->thread_idx])));
				//}

				// Timming -----------------------------------------------------
				timming_end(">> Random MCT Time", ts_mct);
				// Timming -----------------------------------------------------

				if (DEBUG_DEV) validate_solution(&(thread_instance->population[thread_instance->thread_idx]));
			}
	            
            // Espero a que los demas hilos terminen.
            rc = pthread_barrier_wait(thread_instance->sync_barrier);
            if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
            {
                printf("Could not wait on barrier\n");
                exit(EXIT_FAILURE);
            }

			// Comienza la búsqueda.
			pthread_mutex_lock(thread_instance->work_type_mutex);
				thread_instance->work_type[0] = PALS_CPU_1POP_WORK__SEARCH;
			pthread_mutex_unlock(thread_instance->work_type_mutex);
			
        } else if (work_type == PALS_CPU_1POP_WORK__SEARCH) {
            // PALS_CPU_1POP_WORK__SEARCH ====================================================================
           
            // Sorteo la solución con la que me toca trabajar  =====================================================
			#ifdef CPU_MERSENNE_TWISTER
            double random = cpu_mt_generate(*(thread_instance->thread_random_state));
            #else
            double random = cpu_rand_generate(*(thread_instance->thread_random_state));
            #endif
            
            pthread_mutex_lock(thread_instance->population_mutex);
				int random_sol_index = (int)floor(random * (*(thread_instance->population_count)));
				
				if (DEBUG_DEV) {
					fprintf(stdout, "[DEBUG] Random selection\n");
					fprintf(stdout, "        Population_count: %d\n", *(thread_instance->population_count));
					fprintf(stdout, "        Random          : %f\n", random);
					fprintf(stdout, "        Random_sol_index: %d\n", random_sol_index);
				}
				
				int current_sol_pos = -1;
				int current_sol_index = -1;
				
				for (; (current_sol_pos < thread_instance->population_max_size) 
					&& (current_sol_index < *(thread_instance->population_count))
					&& (current_sol_index != random_sol_index); ) {
					
					current_sol_pos++;
					
					if (thread_instance->population[current_sol_pos].status > SOLUTION__STATUS_EMPTY) {
						current_sol_index++;
					}
				}
            pthread_mutex_unlock(thread_instance->population_mutex);
            
			// Determino la estrategia de busqueda del hilo  =====================================================
			int search_type;
			
			#ifdef CPU_MERSENNE_TWISTER
			random = cpu_mt_generate(*(thread_instance->thread_random_state));
			#else
			random = cpu_rand_generate(*(thread_instance->thread_random_state));
			#endif
			
			if (random < PALS_CPU_1POP_SEARCH_BALANCE__MAKESPAN) {
				search_type = PALS_CPU_1POP_SEARCH__MAKESPAN_GREEDY;
				thread_instance->total_makespan_greedy_searches++;
				
			} else if (random < PALS_CPU_1POP_SEARCH_BALANCE__MAKESPAN + PALS_CPU_1POP_SEARCH_BALANCE__ENERGY) {
				search_type = PALS_CPU_1POP_SEARCH__ENERGY_GREEDY;
				thread_instance->total_energy_greedy_searches++;
				
			} else {
				search_type = PALS_CPU_1POP_SEARCH__RANDOM_GREEDY;
				thread_instance->total_random_greedy_searches++;
			}

            // Clono la solución elegida =====================================================
           
            struct solution *selected_solution;
            int selected_solution_pos = -1;
            
            pthread_mutex_lock(thread_instance->population_mutex);
				for (int i = 0; (i < thread_instance->population_max_size) && (selected_solution_pos == -1); i++) {
					if (thread_instance->population[i].status == SOLUTION__STATUS_EMPTY) {
						thread_instance->population[i].status = SOLUTION__STATUS_NOT_READY;
						selected_solution_pos = i;
						if (DEBUG_DEV) fprintf(stdout, "[DEBUG] Found individual %d free\n", selected_solution_pos);
					}
				}

				if (selected_solution_pos >= 0) {
					selected_solution = &(thread_instance->population[selected_solution_pos]);
					
					if (selected_solution->initialized == 0) {
						if (DEBUG_DEV) fprintf(stdout, "[DEBUG] Initializing individual %d\n", selected_solution_pos);
						init_empty_solution(thread_instance->etc, thread_instance->energy, selected_solution);
					}
					
					if (DEBUG_DEV) fprintf(stdout, "[DEBUG] Cloning individual %d to %d\n", current_sol_pos, selected_solution_pos);
					clone_solution(selected_solution, &(thread_instance->population[current_sol_pos]), 0);				
				}
            pthread_mutex_unlock(thread_instance->population_mutex);
							
			if (DEBUG_DEV) {
				fprintf(stdout, "[DEBUG] Selected individual\n");
				fprintf(stdout, "        Original_solutiol_pos = %d\n", current_sol_pos);
				fprintf(stdout, "        Selected_solution_pos = %d\n", selected_solution_pos);
				fprintf(stdout, "        Selected_solution.status = %d\n", selected_solution->status);
				fprintf(stdout, "        Selected_solution.initializd = %d\n", selected_solution->initialized);
			}
			
			int solution_improved_on = 0;
			
			for (int search_iteration = 0; (search_iteration < PALS_CPU_1POP_WORK__THREAD_ITERATIONS) 
				&& (search_iteration - solution_improved_on < PALS_CPU_1POP_WORK__THREAD_CONVERGENCE); search_iteration++) {
							
				thread_instance->total_iterations++;                
			  
				// Determino que tipo movimiento va a realizar el hilo.
				#ifdef CPU_MERSENNE_TWISTER
				random = cpu_mt_generate(*(thread_instance->thread_random_state));
				#else
				random = cpu_rand_generate(*(thread_instance->thread_random_state));
				#endif
				
				int mov_type = PALS_CPU_1POP_SEARCH_OP__SWAP;
				
				if (random < PALS_CPU_1POP_SEARCH_OP_BALANCE__SWAP) {
					mov_type = PALS_CPU_1POP_SEARCH_OP__SWAP;
				} else if (random < PALS_CPU_1POP_SEARCH_OP_BALANCE__MOVE) {
					mov_type = PALS_CPU_1POP_SEARCH_OP__MOVE;
				}

				// Determino las máquinas de inicio para la búsqueda.
				int machine_a, machine_b;

				if (search_type == PALS_CPU_1POP_SEARCH__MAKESPAN_GREEDY) {
					// La estrategia es mejorar makespan, siempre selecciono la máquina que define el makespan.
					machine_a = get_worst_ct_machine_id(selected_solution);
					
				} else if (search_type == PALS_CPU_1POP_SEARCH__ENERGY_GREEDY) {
					// La estrategia es mejorar energía, siempre selecciono la máquina que consume más energía.
					machine_a = get_worst_energy_machine_id(selected_solution);
					
				} else {
					#ifdef CPU_MERSENNE_TWISTER
					random = cpu_mt_generate(*(thread_instance->thread_random_state));
					#else
					random = cpu_rand_generate(*(thread_instance->thread_random_state));
					#endif
					
					// La estrategia es aleatoria.
					machine_a = (int)floor(random * thread_instance->etc->machines_count);
						
				}
				
				#ifdef CPU_MERSENNE_TWISTER
				random = cpu_mt_generate(*(thread_instance->thread_random_state));
				#else
				random = cpu_rand_generate(*(thread_instance->thread_random_state));
				#endif
				
				// Siempre selecciono la segunda máquina aleatoriamente.
				machine_b = (int)floor(random * (thread_instance->etc->machines_count - 1));
					
				if (machine_a == machine_b) machine_b++;

				// Determino las tareas de inicio para la búsqueda.
				int task_x;
				int machine_a_task_count = get_machine_tasks_count(selected_solution, machine_a);
				
				#ifdef CPU_MERSENNE_TWISTER
				random = cpu_mt_generate(*(thread_instance->thread_random_state));
				#else
				random = cpu_rand_generate(*(thread_instance->thread_random_state));
				#endif
				
				task_x = (int)floor(random * machine_a_task_count);

				float machine_a_energy_idle = get_energy_idle_value(thread_instance->energy, machine_a);
				float machine_a_energy_max = get_energy_max_value(thread_instance->energy, machine_a);
				float machine_b_energy_idle = get_energy_idle_value(thread_instance->energy, machine_b);
				float machine_b_energy_max = get_energy_max_value(thread_instance->energy, machine_b);

				float machine_a_ct_old, machine_b_ct_old;
				float machine_a_ct_new, machine_b_ct_new;
				
				float delta_makespan, delta_ct, delta_energy;
				
				int search_ended;
				search_ended = 0;
				
				int task_x_best_move_pos = -1, machine_b_best_move_id = -1;
				int task_x_best_swap_pos = -1, task_y_best_swap_pos = -1;
					
				float current_makespan = get_makespan(selected_solution);
					
				if (mov_type == PALS_CPU_1POP_SEARCH_OP__SWAP) {
					int machine_b_task_count = get_machine_tasks_count(selected_solution, machine_b);
					
					#ifdef CPU_MERSENNE_TWISTER
					random = cpu_mt_generate(*(thread_instance->thread_random_state));
					#else
					random = cpu_rand_generate(*(thread_instance->thread_random_state));
					#endif
					
					int task_y = (int)floor(random * machine_b_task_count);
					
					int top_task_a = PALS_CPU_1POP_WORK__SRC_TASK_NHOOD;
					if (top_task_a > machine_a_task_count) top_task_a = machine_a_task_count;
					
					int top_task_b = PALS_CPU_1POP_WORK__DST_TASK_NHOOD;
					if (top_task_b > machine_b_task_count) top_task_b = machine_b_task_count;
					
					int task_x_pos, task_y_pos;	                
					int task_x_current, task_y_current;
					
					for (int task_x_offset = 0; (task_x_offset < top_task_a) && (search_ended == 0); task_x_offset++) {
						task_x_pos = (task_x + task_x_offset) % machine_a_task_count;
						task_x_current = get_machine_task_id(selected_solution, machine_a, task_x_pos);
					
						for (int task_y_offset = 0; (task_y_offset < top_task_b) && (search_ended == 0); task_y_offset++) {
							task_y_pos = (task_y + task_y_offset) % machine_b_task_count;                        
							task_y_current = get_machine_task_id(selected_solution, machine_b, task_y_pos);
							
							// Máquina 1.
							machine_a_ct_old = get_machine_compute_time(selected_solution, machine_a);
	
							machine_a_ct_new = machine_a_ct_old;
							machine_a_ct_new = machine_a_ct_new - get_etc_value(thread_instance->etc, machine_a, task_x_current); // Resto del ETC de x en a.
							machine_a_ct_new = machine_a_ct_new + get_etc_value(thread_instance->etc, machine_a, task_y_current); // Sumo el ETC de y en a.

							// Máquina 2.
							machine_b_ct_old = get_machine_compute_time(selected_solution, machine_b);

							machine_b_ct_new = machine_b_ct_old;
							machine_b_ct_new = machine_b_ct_new - get_etc_value(thread_instance->etc, machine_b, task_y_current); // Resto el ETC de y en b.
							machine_b_ct_new = machine_b_ct_new + get_etc_value(thread_instance->etc, machine_b, task_x_current); // Sumo el ETC de x en b.
													
							if (search_type == PALS_CPU_1POP_SEARCH__MAKESPAN_GREEDY) {
								// La estrategia es mejorar makespan.
								
								if ((machine_a_ct_new < machine_a_ct_old) && (machine_b_ct_new < machine_a_ct_old)) {
									search_ended = 1;
									task_x_best_swap_pos = task_x_pos;
									task_y_best_swap_pos = task_y_pos;
								}
							} else if (search_type == PALS_CPU_1POP_SEARCH__ENERGY_GREEDY) {
								// La estrategia es mejorar energía.                                                              
								if ((machine_a_ct_new <= current_makespan) && (machine_b_ct_new <= current_makespan)) {
									delta_energy = 
										((machine_a_ct_old - machine_a_ct_new) * (machine_a_energy_max - machine_a_energy_idle)) +
										((machine_b_ct_old - machine_b_ct_new) * (machine_b_energy_max - machine_b_energy_idle));                    
										
									if (delta_energy > 0) {
										/*fprintf(stdout, "[DEBUG] machine_a delta_ct = %f con energy_diff = %f\n",
											(machine_a_ct_old - machine_a_ct_new), (machine_a_energy_max - machine_a_energy_idle));
										fprintf(stdout, "[DEBUG] machine_b delta_ct = %f con energy_diff = %f\n",
											(machine_b_ct_old - machine_b_ct_new), (machine_b_energy_max - machine_b_energy_idle));
										fprintf(stdout, "[DEBUG] delta_energy = %f\n", delta_energy);*/
									
										search_ended = 1;
										task_x_best_swap_pos = task_x_pos;
										task_y_best_swap_pos = task_y_pos;
									}
								}
							} else {
								// La estrategia es aleatoria.
								if ((machine_a_ct_new <= current_makespan) && (machine_b_ct_new <= current_makespan)) {
									delta_ct = (machine_a_ct_old - machine_a_ct_new) + (machine_b_ct_old - machine_b_ct_new);	                        
									
									delta_energy = 
										((machine_a_ct_old - machine_a_ct_new) * (machine_a_energy_max - machine_a_energy_idle)) +
										((machine_b_ct_old - machine_b_ct_new) * (machine_b_energy_max - machine_b_energy_idle));     
										
									if ((delta_energy > 0) || (delta_ct > 0)) {
										search_ended = 1;
										task_x_best_swap_pos = task_x_pos;
										task_y_best_swap_pos = task_y_pos;
									}
								}
							}
						}
					}

				} else if (mov_type == PALS_CPU_1POP_SEARCH_OP__MOVE) {
					int top_task_a = PALS_CPU_1POP_WORK__SRC_TASK_NHOOD;
					if (top_task_a > machine_a_task_count) top_task_a = machine_a_task_count;
					
					int top_machine_b = PALS_CPU_1POP_WORK__DST_MACH_NHOOD;
					if (top_machine_b > thread_instance->etc->machines_count) top_machine_b = thread_instance->etc->machines_count;
					
					int task_x_pos;
					int task_x_current, machine_b_current;
					
					for (int task_x_offset = 0; (task_x_offset < top_task_a) && (search_ended == 0); task_x_offset++) {
						task_x_pos = (task_x + task_x_offset) % machine_a_task_count;
						task_x_current = get_machine_task_id(selected_solution, machine_a, task_x_pos);
					
						for (int machine_b_offset = 0; (machine_b_offset < top_machine_b) && (search_ended == 0); machine_b_offset++) {
							machine_b_current = (machine_b + machine_b_offset) % thread_instance->etc->machines_count;
															
							if (machine_b_current != machine_a) {
								// Máquina 1.
								machine_a_ct_old = get_machine_compute_time(selected_solution, machine_a);
	
								machine_a_ct_new = machine_a_ct_old;
								machine_a_ct_new = machine_a_ct_new - get_etc_value(thread_instance->etc, machine_a, task_x_current); // Resto del ETC de x en a.

								// Máquina 2.
								machine_b_ct_old = get_machine_compute_time(selected_solution, machine_b);

								machine_b_ct_new = machine_b_ct_old;
								machine_b_ct_new = machine_b_ct_new + get_etc_value(thread_instance->etc, machine_b, task_x_current); // Sumo el ETC de x en b.
														
								if (search_type == PALS_CPU_1POP_SEARCH__MAKESPAN_GREEDY) {
									// La estrategia es mejorar makespan.
									delta_makespan = (machine_a_ct_old - machine_b_ct_new);
									
									if (delta_makespan > 0) {
										search_ended = 1;
										task_x_best_move_pos = task_x_pos;
										machine_b_best_move_id = machine_b_current;
									}
								} else if (search_type == PALS_CPU_1POP_SEARCH__ENERGY_GREEDY) {
									// La estrategia es mejorar energía.
									if ((machine_a_ct_new <= current_makespan) && (machine_b_ct_new <= current_makespan)) {
										delta_energy = 
											((machine_a_ct_old - machine_a_ct_new) * (machine_a_energy_max - machine_a_energy_idle)) +
											((machine_b_ct_old - machine_b_ct_new) * (machine_b_energy_max - machine_b_energy_idle));                    
											
										if (delta_energy > 0) {
											/*fprintf(stdout, "[DEBUG] machine_a delta_ct = %f con energy_diff = %f\n",
												(machine_a_ct_old - machine_a_ct_new), (machine_a_energy_max - machine_a_energy_idle));
											fprintf(stdout, "[DEBUG] machine_b delta_ct = %f con energy_diff = %f\n",
												(machine_b_ct_old - machine_b_ct_new), (machine_b_energy_max - machine_b_energy_idle));
											fprintf(stdout, "[DEBUG] delta_energy = %f\n", delta_energy);*/
											
											search_ended = 1;
											task_x_best_move_pos = task_x_pos;
											machine_b_best_move_id = machine_b_current;
										}
									}
								} else {
									// La estrategia es aleatoria.
									if ((machine_a_ct_new <= current_makespan) && (machine_b_ct_new <= current_makespan)) {
										delta_ct = (machine_a_ct_old - machine_a_ct_new) + (machine_b_ct_old - machine_b_ct_new);	                        
										
										delta_energy = 
											((machine_a_ct_old - machine_a_ct_new) * (machine_a_energy_max - machine_a_energy_idle)) +
											((machine_b_ct_old - machine_b_ct_new) * (machine_b_energy_max - machine_b_energy_idle));     
											
										if ((delta_energy > 0) || (delta_ct > 0)) {
											search_ended = 1;
											task_x_best_move_pos = task_x_pos;
											machine_b_best_move_id = machine_b_current;
										}
									}
								}
							}
						}
					}
				}
				
				if (search_ended == 1) {
					solution_improved_on = search_iteration;
					
					// Hago los cambios ======================================================================================
					if (mov_type == PALS_CPU_1POP_SEARCH_OP__SWAP) {
						// Intercambio las tareas!
						if (DEBUG_DEV) fprintf(stdout, "[DEBUG] Ejecuto un SWAP! (%d, %d, %d, %d)\n", machine_a, task_x_best_swap_pos, machine_b, task_y_best_swap_pos);
						swap_tasks_by_pos(selected_solution, machine_a, task_x_best_swap_pos, machine_b, task_y_best_swap_pos);                   
						
						thread_instance->total_swaps++;
					} if (mov_type == PALS_CPU_1POP_SEARCH_OP__MOVE) {
						// Muevo la tarea!
						if (DEBUG_DEV) fprintf(stdout, "[DEBUG] Ejecuto un MOVE! (%d, %d, %d)\n", machine_a, task_x_best_move_pos, machine_b_best_move_id);
						move_task_to_machine_by_pos(selected_solution, machine_a, task_x_best_move_pos, machine_b_best_move_id);
						
						thread_instance->total_moves++;
					}
					
					if (DEBUG_DEV) validate_solution(selected_solution);
				}
			}
			
			// Refresco la energía porque a veces encuentro diferencias. (¿rendondeo?)
			refresh_energy(selected_solution);
		
			// Dejo pronto el nuevo individuo para ser usado.
			pthread_mutex_lock(thread_instance->population_mutex);
				
				int is_non_dominated = pals_cpu_1pop_eval_new_solution(thread_instance, selected_solution_pos);
				if (is_non_dominated == 1) {
					thread_instance->population[selected_solution_pos].status = SOLUTION__STATUS_READY;
					thread_instance->population_count[0] = thread_instance->population_count[0] + 1;
				} else {
					thread_instance->population[selected_solution_pos].status = SOLUTION__STATUS_EMPTY;
				}
							   
			pthread_mutex_unlock(thread_instance->population_mutex);
			
			if (DEBUG_DEV) {
				fprintf(stdout, "[DEBUG] Cantidad de individuos en la población: %d\n", *(thread_instance->population_count));
				validate_thread_instance(thread_instance);
			}
		}
		
		clock_gettime(CLOCK_REALTIME, &ts_current);
    } 

    if (DEBUG_DEV) fprintf(stdout, "[DEBUG] Me mandaron a terminar o se acabó el tiempo! Tengo algo para hacer?\n");

	return NULL;
}
