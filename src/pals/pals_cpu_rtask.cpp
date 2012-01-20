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

#include "pals_cpu_rtask.h"

void pals_cpu_rtask(struct params &input, struct etc_matrix *etc, struct energy_matrix *energy) {
	// ==============================================================================
	// PALS aleatorio por tarea.
	// ==============================================================================
	
	// Timming -----------------------------------------------------
	timespec ts_init;
	timming_start(ts_init);
	// Timming -----------------------------------------------------

	// Inicializo la memoria y los hilos de ejecución.
	struct pals_cpu_rtask_instance instance;
	pals_cpu_rtask_init(input, etc, energy, input.seed, instance);
    
	// Timming -----------------------------------------------------
	timming_end(">> pals_cpu_rtask_init", ts_init);
	// Timming -----------------------------------------------------

    // Bloqueo la ejecución hasta que terminen todos los hilos.
    if(pthread_join(instance.master_thread, NULL))
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
	
	// ===========> DEBUG
	if (DEBUG) {
	    // TODO:................
		/*fprintf(stdout, "[INFO] Cantidad de iteraciones        : %d\n", instance.total_iterations);
		fprintf(stdout, "[INFO] Último mejor encontrado en     : %d\n", instance.last_elite_found_on_iter);
		fprintf(stdout, "[INFO] Total de makespan searches     : %d\n", instance.total_makespan_greedy_searches);
		fprintf(stdout, "[INFO] Total de energy searches       : %d\n", instance.total_energy_greedy_searches);
		fprintf(stdout, "[INFO] Total de random searches       : %d\n", instance.total_random_greedy_searches);
		fprintf(stdout, "[INFO] Total de swaps                 : %ld\n", instance.total_swaps);
		fprintf(stdout, "[INFO] Total de moves                 : %ld\n", instance.total_moves);
		fprintf(stdout, "[INFO] Cantidad de soluciones ND      : %d\n", instance.elite_population_count);
	    */
	    for (int i = 0; i < instance.elite_population_count; i++) {
		    validate_solution(instance.elite_population[i]);
		}
	}
	// <=========== DEBUG
	
	if (DEBUG) {	
	    for (int i = 0; i < instance.elite_population_count; i++) {
    		fprintf(stdout, "[DEBUG] Solution %d: %f | %f\n", i, get_makespan(instance.elite_population[i]), 
    		    get_energy(instance.elite_population[i]));
	    }
	} else {
	        if (!OUTPUT_SOLUTION) { 
        	    for (int i = 0; i < instance.elite_population_count; i++) {
    	            fprintf(stdout, "%f %f\n", get_makespan(instance.elite_population[i]), get_energy(instance.elite_population[i]));
    	        }
	        } else {
				/*for (int task_id = 0; task_id < instance.etc->tasks_count; task_id++) {
					fprintf(stdout, "%d\n", get_task_assigned_machine_id(current_solution,task_id));
				}*/
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

	// Libero la memoria del dispositivo.
	pals_cpu_rtask_finalize(instance);
	
	// Timming -----------------------------------------------------
	timming_end(">> pals_cpu_rtask_finalize", ts_finalize);
	// Timming -----------------------------------------------------		
}

void pals_cpu_rtask_init(struct params &input, struct etc_matrix *etc, struct energy_matrix *energy,
    int seed, struct pals_cpu_rtask_instance &empty_instance) {
    
	// Asignación del paralelismo del algoritmo.
	empty_instance.count_threads = input.thread_count;
	
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
    
    empty_instance.etc = etc;
    empty_instance.energy = energy;

	// Population.
    empty_instance.population = (struct solution*)malloc(sizeof(struct solution) * PALS_CPU_RTASK_WORK__POP_MAX_SIZE);
    if (empty_instance.population == NULL) {
		fprintf(stderr, "[ERROR] Solicitando memoria para population.\n");
		exit(EXIT_FAILURE);
	}
        
    for (int i = 0; i < PALS_CPU_RTASK_WORK__POP_MAX_SIZE; i++) {
        empty_instance.population[i].status = SOLUTION__STATUS_EMPTY;
        empty_instance.population[i].initialized = 0;
    }

    empty_instance.population_locked = (int*)malloc(sizeof(int) * PALS_CPU_RTASK_WORK__POP_MAX_SIZE);
    if (empty_instance.population_locked == NULL) {
		fprintf(stderr, "[ERROR] Solicitando memoria para population locked.\n");
		exit(EXIT_FAILURE);
	}
    
	memset(empty_instance.population_locked, 0, sizeof(int) * PALS_CPU_RTASK_WORK__POP_MAX_SIZE);

	empty_instance.population_count = 0;
	
	// Elite population.
    empty_instance.elite_population = (struct solution**)malloc(sizeof(struct solution*) * PALS_CPU_RTASK_WORK__ELITE_POP_MAX_SIZE);
    if (empty_instance.elite_population == NULL) {
		fprintf(stderr, "[ERROR] Solicitando memoria para elite population.\n");
		exit(EXIT_FAILURE);
	}
           
	memset(empty_instance.elite_population, 0, sizeof(struct solution*) * PALS_CPU_RTASK_WORK__ELITE_POP_MAX_SIZE);
	empty_instance.elite_population_count = 0;
	
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
	
	timming_end(".. cpu_rand_buffers", ts_1);
	
	// =========================================================================
	// Creo e inicializo los threads y los mecanismos de sincronización del sistema.
	
	timespec ts_threads;
	timming_start(ts_threads);

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
	empty_instance.slave_threads = (pthread_t*)
	    malloc(sizeof(pthread_t) * (empty_instance.count_threads - 1));
	
	empty_instance.slave_threads_args = (struct pals_cpu_rtask_thread_arg*)
	    malloc(sizeof(struct pals_cpu_rtask_thread_arg) * (empty_instance.count_threads - 1));

    empty_instance.slave_work_type = (int*)malloc(sizeof(int) * (empty_instance.count_threads - 1));
	for (int i = 0; i < empty_instance.count_threads - 1; i++) {
		if (i < PALS_CPU_RTASK_WORK__POP_MAX_SIZE) {
			empty_instance.slave_work_type[i] = PALS_CPU_RTASK_WORK__INIT_POP;
		} else {
			empty_instance.slave_work_type[i] = PALS_CPU_RTASK_WORK__WAIT;
		}
	}
	
    // Creo el hilo master.
    if (pthread_create(&(empty_instance.master_thread), NULL, pals_cpu_rtask_master_thread, (void*) &(empty_instance)))
    {
        printf("Could not create master thread\n");
        exit(EXIT_FAILURE);
    }
	
	// Creo los hilos esclavos.
	for (int i = 0; i < (empty_instance.count_threads - 1); i++) {
   		empty_instance.slave_threads_args[i].thread_idx = i;
   		
        empty_instance.slave_threads_args[i].etc = empty_instance.etc;
        empty_instance.slave_threads_args[i].energy = empty_instance.energy;
        
        empty_instance.slave_threads_args[i].population = empty_instance.population;
        empty_instance.slave_threads_args[i].population_locked = empty_instance.population_locked;
        empty_instance.slave_threads_args[i].population_count = &(empty_instance.population_count);
        
    	empty_instance.slave_threads_args[i].work_type = &(empty_instance.slave_work_type[i]);
		
        empty_instance.slave_threads_args[i].population_mutex = &(empty_instance.population_mutex);
		empty_instance.slave_threads_args[i].new_solutions_sem = &(empty_instance.new_solutions_sem);
		empty_instance.slave_threads_args[i].sync_barrier = &(empty_instance.sync_barrier);
        	
        empty_instance.slave_threads_args[i].thread_random_state = &(empty_instance.random_states[i]);
   		
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
    free(instance.population);
    free(instance.population_locked);
    
    free(instance.random_states);
        
	free(instance.slave_threads);	
	free(instance.slave_threads_args);
	free(instance.slave_work_type);
	
	pthread_mutex_destroy(&(instance.population_mutex));
	pthread_barrier_destroy(&(instance.sync_barrier));
	sem_destroy(&(instance.new_solutions_sem));
}

int pals_cpu_rtask_eval_new_solutions(struct pals_cpu_rtask_instance *instance) {
	int solutions_found = 0;
	int s_idx = 0;
	
	for (int s_pos = 0; (s_pos < PALS_CPU_RTASK_WORK__POP_MAX_SIZE) && (s_idx < instance->population_count); s_pos++) {
		if (instance->population[s_pos].status == SOLUTION__STATUS_NEW) {
		    int is_non_dominated;
		    is_non_dominated = 1;
		
			// Calculo no dominancia del elemento actual y lo agrego a la población elite si corresponde.
			float makespan_candidate, energy_candidate;
			makespan_candidate = get_makespan(&(instance->population[s_pos]));
			energy_candidate = get_energy(&(instance->population[s_pos]));
			
			int elite_s_idx = -1, added_to_elite = 0, end_loop = 0;
			float makespan_elite_sol, energy_elite_sol;
			
			for (int elite_s_pos = 0; (elite_s_pos < PALS_CPU_RTASK_WORK__ELITE_POP_MAX_SIZE) 
			    && (elite_s_idx < instance->elite_population_count) && (end_loop == 0); elite_s_pos++) {
			    
			    if (instance->elite_population[elite_s_pos] != NULL) {
			        elite_s_idx++;
			    
			        makespan_elite_sol = get_makespan(instance->elite_population[elite_s_pos]);
 			        energy_elite_sol = get_makespan(instance->elite_population[elite_s_pos]);
 			        
 			        if ((makespan_candidate < makespan_elite_sol) && (energy_candidate < energy_elite_sol)) {
 			            // Domina a una solución de la elite.
 			            if (added_to_elite == 0) {
 			                added_to_elite = 1;
     			            instance->elite_population[elite_s_pos] = &(instance->population[s_pos]);
 			            } else {
     			            elite_s_idx--;
     			            instance->elite_population_count--;
     			            instance->elite_population[elite_s_pos] = NULL;
     			        }
 			        } else if ((makespan_candidate < makespan_elite_sol) || (energy_candidate < energy_elite_sol)) {
 			            // Es no dominada.
 			            end_loop = 1;
 			        } else if ((makespan_candidate >= makespan_elite_sol) && (energy_candidate >= energy_elite_sol)) {
 			            // Es dominada.
 			            is_non_dominated = 0;
 			            end_loop = 1;
 			        }
			    }
		    }
			
			if (is_non_dominated == 1) {
			    if ((added_to_elite == 0) && (instance->elite_population_count < PALS_CPU_RTASK_WORK__ELITE_POP_MAX_SIZE)) {
			        for (int elite_s_pos = 0; (elite_s_pos < PALS_CPU_RTASK_WORK__ELITE_POP_MAX_SIZE) && (added_to_elite == 0); elite_s_pos++) {
			            
			            if (instance->elite_population[elite_s_pos] == NULL) {
			                added_to_elite = 1;
     			            instance->elite_population_count++;
     			            instance->elite_population[elite_s_pos] = &(instance->population[s_pos]);
			            }
			        }
			    }

                if (added_to_elite != 1) {
                    instance->total_elite_population_full++;
                    fprintf(stderr, "[WARNING] Población elite llena!!!");
    			}
			}

            // Bloqueo la población para actualizar el estado del individuo.
			pthread_mutex_lock(&(instance->population_mutex));
			instance->population[s_pos].status = SOLUTION__STATUS_READY;
            pthread_mutex_unlock(&(instance->population_mutex));
    		
			solutions_found++;
		}
		
		if (instance->population[s_pos].status > SOLUTION__STATUS_EMPTY) {
			s_idx++;
		}
	}
	
	return solutions_found;
}

void* pals_cpu_rtask_master_thread(void *thread_arg) {
    struct pals_cpu_rtask_instance *instance;
    instance = (struct pals_cpu_rtask_instance *)thread_arg;

    int rc;
	
	int increase_depth = 0;
	int convergence_flag = 0;
   
	timespec ts_start, ts_current;
	clock_gettime(CLOCK_REALTIME, &ts_start);
	clock_gettime(CLOCK_REALTIME, &ts_current);

	// ===========================================================================
	// [SYNC POINT] Espero a que los esclavos terminen de inicializar la población.
	rc = pthread_barrier_wait(&(instance->sync_barrier));
	if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
	{
		printf("Could not wait on barrier\n");
		exit(EXIT_FAILURE);
	}
	
    if (instance->count_threads < PALS_CPU_RTASK_WORK__POP_MAX_SIZE) {
		instance->population_count = instance->count_threads;
	} else {
    	instance->population_count = PALS_CPU_RTASK_WORK__POP_MAX_SIZE;
	}
		
	// Calculo no dominancia de la población inicial y separo la población elite.
	pals_cpu_rtask_eval_new_solutions(instance);
	
	// Asigno trabajo de busqueda a todos los hilos.
	for (int i = 0; i < instance->count_threads - 1; i++) {
		instance->slave_work_type[i] = PALS_CPU_RTASK_WORK__SEARCH;
	}

	// [SYNC POINT] Aviso a los esclavos que pueden continuar.
	rc = pthread_barrier_wait(&(instance->sync_barrier));
	if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
	{
		printf("Could not wait on barrier\n");
		exit(EXIT_FAILURE);
	}
	
	int iter;
	for (iter = 0; (convergence_flag == 0) && (ts_current.tv_sec - ts_start.tv_sec < PALS_CPU_RTASK_WORK__TIMEOUT); iter++) {
		// Timming -----------------------------------------------------
		timespec ts_search;
		timming_start(ts_search);
		// Timming -----------------------------------------------------
	
        // ===========================================================================
        // Espero a que un esclavo encuentre una solución.
		
		if (sem_wait(&(instance->new_solutions_sem)) == -1) {
			printf("Could not wait on semaphore\n");
			exit(EXIT_FAILURE);
		}
	
        // ===========================================================================
        // Busco las soluciones nuevas y calculo si son dominadas o no dominadas.
	
		int solutions_found;
		solutions_found = pals_cpu_rtask_eval_new_solutions(instance);
		
		// Si no encuentro soluciones nuevas, aumento el contador de convergencia.
		// De lo contrario lo reinicializo.
		if (solutions_found > 0) {
			increase_depth = 0;
		} else {
			increase_depth++;
		}
	
	    // Después de K iteraciones, o si se llena la población principal, reinicializo la población y dejo
	    // solo los individuos elite.
		if ((iter % PALS_CPU_RTASK_WORK__RESET_POP == 0) || (instance->population_count == PALS_CPU_RTASK_WORK__POP_MAX_SIZE)) {
			// Muevo toda la población elite a la población y elimino el resto. ---------------------------
			
			if (DEBUG) {
			    fprintf(stdout, "[DEBUG] Limpio la población con >> \n");
			    fprintf(stdout, "        Cantidad de iteraciones: %d\n", iter % PALS_CPU_RTASK_WORK__RESET_POP);
			    fprintf(stdout, "        Población              : %d\n", instance->population_count);
			    fprintf(stdout, "        Población elite        : %d\n", instance->elite_population_count);
			}
			
			// Solicito a los hilos que se detengan.
			for (int i = 0; i < instance->count_threads - 1; i++) {
				instance->slave_work_type[i] = PALS_CPU_RTASK_WORK__WAIT;
			}

			// [SYNC POINT] Espero a que los esclavos terminen.
			rc = pthread_barrier_wait(&(instance->sync_barrier));
			if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
			{
				printf("Could not wait on barrier\n");
				exit(EXIT_FAILURE);
			}
			
			// Proceso las soluciones que pudieron haber sido encontradas mientras detenía los threads.
			pals_cpu_rtask_eval_new_solutions(instance);
			
			// Elimino toda la población que no sea elite.
            for (int s_pos = 0; s_pos < PALS_CPU_RTASK_WORK__POP_MAX_SIZE; s_pos++) {
                instance->population[s_pos].status = SOLUTION__STATUS_EMPTY;
            }
            
            // Recorro toda la población elite y los dejo ready.
            for (int s_pos = 0; s_pos < PALS_CPU_RTASK_WORK__ELITE_POP_MAX_SIZE; s_pos++) {
                if (instance->elite_population[s_pos] != NULL) {
                    instance->elite_population[s_pos]->status = SOLUTION__STATUS_READY;                    
                }
            }
            			
			instance->population_count = instance->elite_population_count;
			
			increase_depth = 0;
			
			// Asigno trabajo de busqueda a todos los hilos.
			for (int i = 0; i < instance->count_threads - 1; i++) {
				instance->slave_work_type[i] = PALS_CPU_RTASK_WORK__SEARCH;
			}

			// [SYNC POINT] Aviso a los esclavos que pueden continuar.
			rc = pthread_barrier_wait(&(instance->sync_barrier));
			if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
			{
				printf("Could not wait on barrier\n");
				exit(EXIT_FAILURE);
			}
		}
                
		// Timming -----------------------------------------------------
		timming_end(">> pals_cpu_rtask_search", ts_search);
		// Timming -----------------------------------------------------

		if (increase_depth >= PALS_CPU_RTASK_WORK__CONVERGENCE) {
			convergence_flag = 1;
		}
		
		clock_gettime(CLOCK_REALTIME, &ts_current);
	}
	
    // Mando a los esclavos a terminar.
    for (int i = 0; i < instance->count_threads - 1; i++) {
        instance->slave_work_type[i] = PALS_CPU_RTASK_WORK__EXIT;
    }

    rc = pthread_barrier_wait(&(instance->sync_barrier));
    if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
    {
        printf("Could not wait on barrier\n");
        exit(EXIT_FAILURE);
    }
        	
	return NULL;
}

void* pals_cpu_rtask_slave_thread(void *thread_arg) {	
    int rc;

    pals_cpu_rtask_thread_arg *thread_instance;
    thread_instance = (pals_cpu_rtask_thread_arg*)thread_arg;

	thread_instance->total_iterations = 0;
    thread_instance->total_makespan_greedy_searches = 0;
    thread_instance->total_energy_greedy_searches = 0;
    thread_instance->total_random_greedy_searches = 0;
	thread_instance->total_swaps = 0;
    thread_instance->total_moves = 0;
    thread_instance->total_population_full = 0;
	   
    while (*(thread_instance->work_type) != PALS_CPU_RTASK_WORK__EXIT) {
        if (*(thread_instance->work_type) == PALS_CPU_RTASK_WORK__WAIT) {
            // PALS_CPU_RTASK_WORK__DO_WAIT ====================================================================
        
            // Espero a que el master me permita seguir.
            rc = pthread_barrier_wait(thread_instance->sync_barrier);
            if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
            {
                printf("Could not wait on barrier\n");
                exit(EXIT_FAILURE);
            }
        } else if (*(thread_instance->work_type) == PALS_CPU_RTASK_WORK__INIT_POP) {
            // PALS_CPU_RTASK_WORK__INIT_POP ===================================================================

            // Inicializo el individuo que me tocó.

	        // Timming -----------------------------------------------------
	        timespec ts_mct;
	        timming_start(ts_mct);
	        // Timming -----------------------------------------------------

            init_empty_solution(thread_instance->etc, thread_instance->energy, &(thread_instance->population[thread_instance->thread_idx]));
            
            int random_task = (int)floor(cpu_rand_generate(*(thread_instance->thread_random_state)) * thread_instance->etc->tasks_count);
	        compute_custom_mct(&(thread_instance->population[thread_instance->thread_idx]), random_task);

            thread_instance->population[thread_instance->thread_idx].status = SOLUTION__STATUS_NEW;

	        // Timming -----------------------------------------------------
	        timming_end(">> Random MCT Time", ts_mct);
	        // Timming -----------------------------------------------------

	        if (DEBUG) validate_solution(&(thread_instance->population[thread_instance->thread_idx]));
	            
            // Espero a que los demas esclavos terminen.
            rc = pthread_barrier_wait(thread_instance->sync_barrier);
            if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
            {
                printf("Could not wait on barrier\n");
                exit(EXIT_FAILURE);
            }    

            // Listo! Población inicializada. Dejo que el master trabaje y espero a que me asigne trabajo.
            
            // Espero a que el master asigne trabajo.
            rc = pthread_barrier_wait(thread_instance->sync_barrier);
            if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
            {
                printf("Could not wait on barrier\n");
                exit(EXIT_FAILURE);
            }
        } else if (*(thread_instance->work_type) == PALS_CPU_RTASK_WORK__SEARCH) {
            // PALS_CPU_RTASK_WORK__SEARCH ====================================================================

            int solution_selected;
            solution_selected = 0;
         
            // Sorteo la solución con la que me toca trabajar.
            pthread_mutex_lock(thread_instance->population_mutex);
            
            int random_sol_index = (int)floor(cpu_rand_generate(*(thread_instance->thread_random_state)) 
                * (double)*(thread_instance->population_count));
            
            int current_sol_pos = 0;
            int current_sol_index = -1;
            
            for (; (current_sol_pos < PALS_CPU_RTASK_WORK__POP_MAX_SIZE) 
                && (current_sol_index < *(thread_instance->population_count))
                && (current_sol_index != random_sol_index); current_sol_pos++) {
                
                if (thread_instance->population[current_sol_pos].status > SOLUTION__STATUS_EMPTY) {
                    current_sol_index++;
                }
            }
            
            // Continúo solo si la tarea sorteada no estaba marcada para eliminar.
            if (thread_instance->population[current_sol_pos].status != SOLUTION__STATUS_TO_DEL)  {
                thread_instance->population_locked[current_sol_pos] = thread_instance->population_locked[current_sol_pos] + 1;
                solution_selected = 1;
            }
                        
            pthread_mutex_unlock(thread_instance->population_mutex);
            
            if (solution_selected == 1) {
                // Determino la estrategia de busqueda del hilo.
               	int search_type;
               	if (cpu_rand_generate(*(thread_instance->thread_random_state)) < 0.33) {
                    search_type = PALS_CPU_RTASK_SEARCH__MAKESPAN_GREEDY;
                    thread_instance->total_makespan_greedy_searches++;
                    
               	} else if (cpu_rand_generate(*(thread_instance->thread_random_state)) < 0.66) {
                    search_type = PALS_CPU_RTASK_SEARCH__ENERGY_GREEDY;
                    thread_instance->total_energy_greedy_searches++;
                    
                } else {
                    search_type = PALS_CPU_RTASK_SEARCH__RANDOM_GREEDY;
                    thread_instance->total_random_greedy_searches++;
               	}

                struct solution *selected_solution;
                selected_solution = &(thread_instance->population[current_sol_pos]);
                
                int is_solution_cloned = 0;
                int solution_improved_on = 0;
                
                for (int search_iteration = 0; (search_iteration < PALS_CPU_RTASK_WORK__THREAD_ITERATIONS) 
                    && (search_iteration - solution_improved_on < PALS_CPU_RTASK_WORK__THREAD_CONVERGENCE); search_iteration++) {
                                
                    thread_instance->total_iterations++;                
                  
                    // Determino que tipo movimiento va a realizar el hilo.
                   	int mov_type;
                   	if (cpu_rand_generate(*(thread_instance->thread_random_state)) < 0.75) {
                   	    mov_type = PALS_CPU_RTASK_SEARCH_OP__SWAP;
                   	} else {
                   	    mov_type = PALS_CPU_RTASK_SEARCH_OP__MOVE;
                   	}

                    // Determino las máquinas de inicio para la búsqueda.
                    int machine_a, machine_b;

                    if (search_type == PALS_CPU_RTASK_SEARCH__MAKESPAN_GREEDY) {
                        // La estrategia es mejorar makespan, siempre selecciono la máquina que define el makespan.
                        machine_a = get_worst_ct_machine_id(selected_solution);
                        
                    } else if (search_type == PALS_CPU_RTASK_SEARCH__ENERGY_GREEDY) {
                        // La estrategia es mejorar energía, siempre selecciono la máquina que consume más energía.
                        machine_a = get_worst_energy_machine_id(selected_solution);
                        
                    } else {
                        // La estrategia es aleatoria.
	                    machine_a = (int)floor(cpu_rand_generate(*(thread_instance->thread_random_state)) * 
	                        thread_instance->etc->machines_count);
	                        
                    }
                    
                    // Siempre selecciono la segunda máquina aleatoriamente.
	                machine_b = (int)floor(cpu_rand_generate(*(thread_instance->thread_random_state)) * 
	                    (thread_instance->etc->machines_count - 1));
	                    
                    if (machine_a == machine_b) machine_b++;

                    // Determino las tareas de inicio para la búsqueda.
                    int task_x;
                    int machine_a_task_count = get_machine_tasks_count(selected_solution, machine_a);
                    
	                task_x = (int)floor(cpu_rand_generate(*(thread_instance->thread_random_state)) * machine_a_task_count);

                    float current_makespan = get_makespan(selected_solution);

                    float machine_a_energy_idle = get_energy_idle_value(thread_instance->energy, machine_a);
                    float machine_a_energy_max = get_energy_max_value(thread_instance->energy, machine_a);
                    float machine_b_energy_idle = get_energy_idle_value(thread_instance->energy, machine_b);
                    float machine_b_energy_max = get_energy_max_value(thread_instance->energy, machine_b);

                    float machine_a_ct_old, machine_b_ct_old;
                    float machine_a_energy_old, machine_b_energy_old;
                    
                    float machine_a_ct_new, machine_b_ct_new;
                    float machine_a_energy_new, machine_b_energy_new;
                    
                    float delta_makespan, delta_ct, delta_energy;
                    
                    int search_ended = 0;

                    if (mov_type == PALS_CPU_RTASK_SEARCH_OP__SWAP) {
                        int machine_b_task_count = get_machine_tasks_count(selected_solution, machine_b);
                        int task_y = (int)floor(cpu_rand_generate(*(thread_instance->thread_random_state)) * machine_b_task_count);
	                    
	                    int top_task_a = PALS_CPU_RTASK_WORK__SRC_TASK_NHOOD;
	                    if (top_task_a > machine_a_task_count) top_task_a = machine_a_task_count;
	                    
	                    int top_task_b = PALS_CPU_RTASK_WORK__DST_TASK_NHOOD;
	                    if (top_task_b > machine_b_task_count) top_task_b = machine_b_task_count;
	                    
	                    int task_x_best_swap_pos, task_y_best_swap_pos;
	                    int task_x_pos, task_y_pos;	                
	                    int task_x_current, task_y_current;
	                    
	                    for (int task_x_offset = 0; (task_x_offset < top_task_a) && (search_ended == 0); task_x_offset++) {
	                        task_x_pos = (task_x + task_x_offset) % machine_a_task_count;
	                        task_x_current = get_machine_task_id(selected_solution, machine_a, task_x_pos);
	                    
	                        for (int task_y_offset = 0; (task_y_offset < top_task_b) && (search_ended == 0); task_y_offset++) {
	                            task_y_pos = (task_y + task_y_offset) % machine_b_task_count;                        
	                            task_y_current = get_machine_task_id(selected_solution, machine_a, task_y_pos);
	                            
	                            // Máquina 1.
	                            machine_a_ct_old = get_machine_compute_time(selected_solution, machine_a);
		
	                            machine_a_ct_new = machine_a_ct_old;
	                            machine_a_ct_new = machine_a_ct_new - get_etc_value(thread_instance->etc, machine_a, task_x_current); // Resto del ETC de x en a.
	                            machine_a_ct_new = machine_a_ct_new + get_etc_value(thread_instance->etc, machine_a, task_y_current); // Sumo el ETC de y en a.

	                            // Máquina 2.
	                            machine_b_ct_old = get_machine_compute_time(selected_solution, machine_b);

	                            machine_b_ct_new = machine_b_ct_old;
	                            machine_b_ct_new = machine_b_ct_new - get_etc_value(thread_instance->etc, machine_b, task_y_current); // Resto el ETC de y en b.
	                            machine_b_ct_new = machine_b_ct_new + get_etc_value(thread_instance->etc, machine_a, task_x); // Sumo el ETC de x en b.
	                                                    
                                if (search_type == PALS_CPU_RTASK_SEARCH__MAKESPAN_GREEDY) {
                                    // La estrategia es mejorar makespan.
                                    delta_makespan = (machine_a_ct_old - machine_a_ct_new) + (machine_a_ct_old - machine_b_ct_new);
                                    
                                    if (delta_makespan > 0) {
                                        search_ended = 1;
                    	                task_x_best_swap_pos = task_x_pos;
                    	                task_y_best_swap_pos = task_x_pos;
	                                }
                                } else if (search_type == PALS_CPU_RTASK_SEARCH__ENERGY_GREEDY) {
                                    // La estrategia es mejorar energía.                                                                
                                    delta_energy = 
                                        ((machine_a_ct_old - machine_a_ct_new) * (machine_a_energy_max - machine_a_energy_idle)) +
                                        ((machine_b_ct_old - machine_b_ct_new) * (machine_b_energy_max - machine_b_energy_idle));                    
                                        
                                    if (delta_energy > 0) {
                                        search_ended = 1;
                    	                task_x_best_swap_pos = task_x_pos;
                    	                task_y_best_swap_pos = task_x_pos;
                    	            }
                                } else {
                                    // La estrategia es aleatoria.
                                    delta_ct = (machine_a_ct_old - machine_a_ct_new) + (machine_b_ct_old - machine_b_ct_new);	                        
                                    
                                    delta_energy = 
                                        ((machine_a_ct_old - machine_a_ct_new) * (machine_a_energy_max - machine_a_energy_idle)) +
                                        ((machine_b_ct_old - machine_b_ct_new) * (machine_b_energy_max - machine_b_energy_idle));     
                                        
                                    if ((delta_energy > 0) || (delta_ct > 0)) {
                                        search_ended = 1;
                    	                task_x_best_swap_pos = task_x_pos;
                    	                task_y_best_swap_pos = task_x_pos;
                                    }
                                }
                            }
                        }

                   	} else if (mov_type == PALS_CPU_RTASK_SEARCH_OP__MOVE) {                
	                    int top_task_a = PALS_CPU_RTASK_WORK__SRC_TASK_NHOOD;
	                    if (top_task_a > machine_a_task_count) top_task_a = machine_a_task_count;
	                    
	                    int top_machine_b = PALS_CPU_RTASK_WORK__DST_MACH_NHOOD;
	                    if (top_machine_b > thread_instance->etc->machines_count) top_machine_b = thread_instance->etc->machines_count;
	                    
	                    int task_x_best_move_pos, machine_b_best_move_id;
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
	                                machine_b_ct_new = machine_b_ct_new + get_etc_value(thread_instance->etc, machine_a, task_x); // Sumo el ETC de x en b.
	                                                        
                                    if (search_type == PALS_CPU_RTASK_SEARCH__MAKESPAN_GREEDY) {
                                        // La estrategia es mejorar makespan.
                                        delta_makespan = (machine_a_ct_old - machine_b_ct_new);
                                        
                                        if (delta_makespan > 0) {
                                            search_ended = 1;
                        	                task_x_best_move_pos = task_x_current;
                        	                machine_b_best_move_id = machine_b_current;
	                                    }
                                    } else if (search_type == PALS_CPU_RTASK_SEARCH__ENERGY_GREEDY) {
                                        // La estrategia es mejorar energía.                                                                
                                        delta_energy = 
                                            ((machine_a_ct_old - machine_a_ct_new) * (machine_a_energy_max - machine_a_energy_idle)) +
                                            ((machine_b_ct_old - machine_b_ct_new) * (machine_b_energy_max - machine_b_energy_idle));                    
                                            
                                        if (delta_energy > 0) {
                                            search_ended = 1;
                        	                task_x_best_move_pos = task_x_current;
                        	                machine_b_best_move_id = machine_b_current;
                        	            }
                                    } else {
                                        // La estrategia es aleatoria.
                                        delta_ct = (machine_a_ct_old - machine_a_ct_new) + (machine_b_ct_old - machine_b_ct_new);	                        
                                        
                                        delta_energy = 
                                            ((machine_a_ct_old - machine_a_ct_new) * (machine_a_energy_max - machine_a_energy_idle)) +
                                            ((machine_b_ct_old - machine_b_ct_new) * (machine_b_energy_max - machine_b_energy_idle));     
                                            
                                        if ((delta_energy > 0) || (delta_ct > 0)) {
                                            search_ended = 1;
                        	                task_x_best_move_pos = task_x_current;
                        	                machine_b_best_move_id = machine_b_current;
                                        }
                                    }
                                }
                   	        }
                   	    }
                   	}
                   	
       	            if (search_ended == 1) {
       	                if (is_solution_cloned == 0) {
           	                // Bloqueo la población para encontrar un lugar donde guardar mi nueva solución.
                            pthread_mutex_lock(thread_instance->population_mutex);

           	                int dst_solution_pos;
           	                int dst_solution_found = 0;            
           	                
           	                for (dst_solution_pos = -1; (dst_solution_pos < (PALS_CPU_RTASK_WORK__POP_MAX_SIZE-1)) && (dst_solution_found == 0); ) {
               	                dst_solution_pos++;
           	                
           	                    if (thread_instance->population_locked[dst_solution_pos] == 0) {
           	                        if ((thread_instance->population[dst_solution_pos].status == SOLUTION__STATUS_EMPTY) ||
           	                            (thread_instance->population[dst_solution_pos].status == SOLUTION__STATUS_TO_DEL)) {

                                        thread_instance->population[dst_solution_pos].status = SOLUTION__STATUS_NOT_READY;
                                        dst_solution_found = 1;
           	                        }
           	                    }  
           	                }
                                        
                            pthread_mutex_unlock(thread_instance->population_mutex);
                            
                            if (dst_solution_found == 1) {
                                if (thread_instance->population[dst_solution_pos].initialized == 0) {
                                    // Si nunca utilicé este espacio, lo inicializo.
                                    init_empty_solution(thread_instance->etc, thread_instance->energy, &(thread_instance->population[dst_solution_pos]));
                                    thread_instance->population[dst_solution_pos].initialized = 1;
                                }
                                
                                // Listo, tengo el espacio libre para mi nuevo individuo. Lo clono.
                                clone_solution(&(thread_instance->population[dst_solution_pos]), selected_solution);      

                                // Ya tengo el clon. Puedo liberar el recurso de lock de la población.
                                pthread_mutex_lock(thread_instance->population_mutex);
                                thread_instance->population_locked[current_sol_pos] = thread_instance->population_locked[current_sol_pos] - 1;
                                pthread_mutex_unlock(thread_instance->population_mutex);
                                
                                // Continúo las iteraciones desde el clon.
                                selected_solution = &(thread_instance->population[dst_solution_pos]);
                                is_solution_cloned = 1;
                            } else {
                                // No enocontré lugar? la población esta llena?
                                // Implemento una politica de reemplazo?
                                thread_instance->total_population_full++;
                                fprintf(stderr, "[WARNING] Población llena!!!");
                            }
                        } 
                        
                        if (is_solution_cloned == 1) {
                            solution_improved_on = search_iteration;
                            
                            if (mov_type == PALS_CPU_RTASK_SEARCH_OP__SWAP) {
                                // TODO: Calcular movimiento!!!!............................                            
                            } if (mov_type == PALS_CPU_RTASK_SEARCH_OP__MOVE) {
                                // TODO: Calcular movimiento!!!!............................
                            }
                        }
                    }
                }
                
                if (is_solution_cloned == 1) {
                    // Dejo pronto el nuevo individuo para ser usado.
                    pthread_mutex_lock(thread_instance->population_mutex);
                    selected_solution->status = SOLUTION__STATUS_NEW;
                    pthread_mutex_unlock(thread_instance->population_mutex);
                }
                                
                // SEMAFOROOOOOOOOOOOOOOOO!!!!
            }
        }  
    } 

	return NULL;
}
