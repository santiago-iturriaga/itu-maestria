#include <pthread.h>
#include <math.h>
#include <stdlib.h>

#include "cmochc_island.h"

#include "cmochc_island_utils.h"
#include "cmochc_island_chc.h"

/* Statistics */
#ifdef DEBUG_1
    int COUNT_GENERATIONS[MAX_THREADS] = {0};
    
    /* Al menos una nueva solución fue agregada a la población de 
     * padres del CHC del deme luego de aplicar una iteración de crossovers */
    int COUNT_AT_LEAST_ONE_CHILDREN_INSERTED[MAX_THREADS] = {0}; 
    /* Cantidad de veces que el CHC del deme mejoró la mejor solución que tenía */
    int COUNT_IMPROVED_BEST_SOL[MAX_THREADS] = {0}; 
    /* Cantidad de crossovers aplicados */
    int COUNT_CROSSOVER[MAX_THREADS] = {0}; 
    /* Cantidad de crossovers que produjeron al menos uno de los hijos 
     * mejor a alguno de sus padres */
    int COUNT_IMPROVED_CROSSOVER[MAX_THREADS] = {0}; 
    /* Cantidad de soluciones mutadas en cataclismos */
    int COUNT_CATACLYSM[MAX_THREADS] = {0}; 
    /* Cantidad de soluciones mejoradas durante cataclismos */
    int COUNT_IMPOVED_CATACLYSM[MAX_THREADS] = {0}; 
    /* Cantidad de ejecuciones del método de migracion */
    int COUNT_MIGRATIONS[MAX_THREADS] = {0}; 
    /* Cantidad de soluciones migradas */
    int COUNT_SOLUTIONS_MIGRATED[MAX_THREADS] = {0}; 
    /* Cantidad histórica de ocurrencias de cada peso en el archivo AGA */
    int COUNT_HISTORIC_WEIGHTS[CMOCHC_PARETO_FRONT__PATCHES] = {0}; 
#endif

struct cmochc_island EA_INSTANCE;

struct cmochc_thread {
    /* Id del esclavo */
    int thread_id;
};

/* Inicializa los hilos y las estructuras de datos */
void init(struct cmochc_thread *threads_data);

/* Obtiene los mejores elementos de cada población */
int gather();

/* Adapta los pesos de los threads */
int adapt_weights_mod_arm(RAND_STATE rstate);

/* Muestra el resultado de la ejecución */
void display_results();

/* Libera los recursos pedidos y finaliza la ejecución */
void finalize();

/* Logica de los esclavos */
void* slave_thread(void *data);

void compute_cmochc_island() {
    if (MAX_THREADS >= INPUT.thread_count) {
        fprintf(stderr, "[ERROR] Max. number of threads is %d (< %d)\n", MAX_THREADS, INPUT.thread_count);
    }
    
    // ==============================================================================
    // CPU CHC (islands)
    // ==============================================================================
    #if defined(DEBUG_0)
        fprintf(stderr, "[DEBUG] starting CPU CHC (islands)...\n");
    #endif

    // Timming -----------------------------------------------------
    TIMMING_START(ts_init);
    // Timming -----------------------------------------------------

    // Inicializo la memoria e inicializo los hilos de ejecucion.
    #if defined(DEBUG_1)
        fprintf(stderr, "[DEBUG] CPU CHC (islands): init\n");
    #endif

    struct cmochc_thread threads[MAX_THREADS];
    init(threads);

    #if defined(DEBUG_1)
        fprintf(stderr, " [OK]\n");
    #endif

    // Timming -----------------------------------------------------
    TIMMING_END(">> cmochc_init", ts_init);
    // Timming -----------------------------------------------------

    int rc;
    int sols_gathered;
    int last_iter_sols_gathered = 0;
    
    RAND_STATE rstate;
    RAND_INIT(INPUT.thread_count, rstate);
       
    for (int iteracion = 0; iteracion < INPUT.max_iterations; iteracion++) {
        /* ************************************************ */
        /* Espero que los esclavos terminen de evolucionar  */
        /* ************************************************ */
        rc = pthread_barrier_wait(&EA_INSTANCE.sync_barrier);
        if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
        {
            printf("Could not wait on barrier\n");
            exit(EXIT_FAILURE);
        }

        /* Incorporo las mejores soluciones al repositorio de soluciones */
        TIMMING_START(ts_gather);
        #if defined(DEBUG_3)
            fprintf(stderr, "[DEBUG] CPU CHC (islands): gather\n");
        #endif

        sols_gathered = gather();
        if (sols_gathered > 0) last_iter_sols_gathered = iteracion;

        #ifndef CMOCHC_PARETO_FRONT__FIXED_WEIGHTS
            #ifdef CMOCHC_PARETO_FRONT__RANDOM_WEIGHTS
                FLOAT random;
            
                for (int i = 0; i < INPUT.thread_count; i++) {
                    random = RAND_GENERATE(rstate);
                    EA_INSTANCE.thread_weight_assignment[i] = (int)(random * CMOCHC_PARETO_FRONT__PATCHES);
                }
            #endif
            #if defined(CMOCHC_PARETO_FRONT__ADAPT_AR_WEIGHTS) || defined(CMOCHC_PARETO_FRONT__ADAPT_AM_WEIGHTS)
                adapt_weights_mod_arm(rstate);
            #endif
        #endif

        TIMMING_END(">> cmochc_gather", ts_gather);

        if (iteracion + 1 >= INPUT.max_iterations) {
            /* Si esta es la úlitma iteracion, les aviso a los esclavos */
            EA_INSTANCE.stopping_condition = 1;
        }

        /* *********************************************************** */
        /* Notifico a los esclavos que terminó la operación de gather  */
        /* *********************************************************** */
        rc = pthread_barrier_wait(&EA_INSTANCE.sync_barrier);
        if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
        {
            printf("Could not wait on barrier\n");
            exit(EXIT_FAILURE);
        }
    }

    RAND_FINALIZE(rstate);

    /* Bloqueo la ejecucion hasta que terminen todos los hilos. */
    for(int i = 0; i < INPUT.thread_count; i++)
    {
        if(pthread_join(EA_INSTANCE.threads[i], NULL))
        {
            fprintf(stderr, "Could not join thread %d\n", i);
            exit(EXIT_FAILURE);
        }
        else
        {
            #if defined(DEBUG_1)
                fprintf(stderr, "[DEBUG] thread %d <OK>\n", i);
            #endif
        }
    }

    /* Libero la memoria. */
    #if defined(DEBUG_1)
        fprintf(stderr, "[DEBUG] CPU CHC (islands): finalize\n");
    #endif

    display_results();
    
    #if defined(DEBUG_1)
        fprintf(stderr, "[DEBUG] Last solution gathered on iteration = %d\n", last_iter_sols_gathered);
    #endif
    
    finalize();
}

int adapt_weights_mod_arm(RAND_STATE rstate) {
    EA_INSTANCE.weight_gap_count = 0;
    int last_filled_patch = -1;
    
    #ifdef DEBUG_3
        fprintf(stderr, "[DEBUG] Archive tag count:\n");
        for (int t = 0; t < CMOCHC_PARETO_FRONT__PATCHES; t++) {
            fprintf(stderr, "> [%d] = %d\n", t, EA_INSTANCE.archiver.tag_count[t]);
        }
    #endif
    
    for (int i = 0; i < CMOCHC_PARETO_FRONT__PATCHES; i++) {
        if (EA_INSTANCE.archiver.tag_count[i] > 0) {
            EA_INSTANCE.weight_gap_index[EA_INSTANCE.weight_gap_count] = i;
            EA_INSTANCE.weight_gap_length[EA_INSTANCE.weight_gap_count] = i - last_filled_patch - 1;
            EA_INSTANCE.weight_gap_sorted[EA_INSTANCE.weight_gap_count] = EA_INSTANCE.weight_gap_count;
            EA_INSTANCE.weight_gap_count++;
            
            last_filled_patch = i;
        }
    }
    
    if (CMOCHC_PARETO_FRONT__PATCHES > last_filled_patch + 1) {
        EA_INSTANCE.weight_gap_index[EA_INSTANCE.weight_gap_count] = CMOCHC_PARETO_FRONT__PATCHES;
        EA_INSTANCE.weight_gap_length[EA_INSTANCE.weight_gap_count] = CMOCHC_PARETO_FRONT__PATCHES - last_filled_patch - 1;
        EA_INSTANCE.weight_gap_sorted[EA_INSTANCE.weight_gap_count] = EA_INSTANCE.weight_gap_count;
        EA_INSTANCE.weight_gap_count++;
    }
    
    #ifdef DEBUG_3
        fprintf(stderr, "[DEBUG] Found gaps:\n");
        for (int i = 0; i < EA_INSTANCE.weight_gap_count; i++) {
            fprintf(stderr, "> [index=%d] pos=%d size=%d\n", i, 
                EA_INSTANCE.weight_gap_index[i], EA_INSTANCE.weight_gap_length[i]);
        }
    #endif
    
    gap_merge_sort();

    #ifdef DEBUG_3
        fprintf(stderr, "[DEBUG] Sorted found gaps:\n");
        for (int i = 0; i < EA_INSTANCE.weight_gap_count; i++) {
            fprintf(stderr, "> [index=%d]<offset=%d> pos=%d size=%d\n", i, EA_INSTANCE.weight_gap_sorted[i],
                EA_INSTANCE.weight_gap_index[EA_INSTANCE.weight_gap_sorted[i]], EA_INSTANCE.weight_gap_length[EA_INSTANCE.weight_gap_sorted[i]]);
        }
    #endif
       
    #if defined(CMOCHC_PARETO_FRONT__ADAPT_AR_WEIGHTS)
        double random;
    #endif
    
    int sel_patch_idx = -1;
    int biggest_patch_index = EA_INSTANCE.weight_gap_sorted[EA_INSTANCE.weight_gap_count - 1];
        
    #ifdef DEBUG_3
        fprintf(stderr, "[DEBUG] biggest_patch_index = %d\n", biggest_patch_index);
    #endif
       
    for (int t = 0; t < INPUT.thread_count; t++) {
        #if defined(CMOCHC_PARETO_FRONT__ADAPT_AR_WEIGHTS)
            if (EA_INSTANCE.weight_gap_length[biggest_patch_index] == 0) {
                sel_patch_idx = EA_INSTANCE.weight_gap_index[biggest_patch_index];
            } else if (EA_INSTANCE.weight_gap_length[biggest_patch_index] == 1) {
                sel_patch_idx = EA_INSTANCE.weight_gap_index[biggest_patch_index] - 1;
            } else {
                random = RAND_GENERATE(rstate);
                
                int random_length;
                random_length = EA_INSTANCE.weight_gap_length[biggest_patch_index] * random;
                
                #ifdef DEBUG_3
                    fprintf(stderr, "[DEBUG] instance.weight_gap_index=%d random_length=%d\n", 
                        EA_INSTANCE.weight_gap_index[biggest_patch_index], random_length);
                #endif
                
                sel_patch_idx = EA_INSTANCE.weight_gap_index[biggest_patch_index] - random_length - 1;
                if (sel_patch_idx < 0) sel_patch_idx = 0;
                if (sel_patch_idx >= CMOCHC_PARETO_FRONT__PATCHES) sel_patch_idx = CMOCHC_PARETO_FRONT__PATCHES - 1;
            }
        #endif
        #if defined(CMOCHC_PARETO_FRONT__ADAPT_AM_WEIGHTS)            
            if (EA_INSTANCE.weight_gap_length[biggest_patch_index] == 0) {
                sel_patch_idx = EA_INSTANCE.weight_gap_index[biggest_patch_index];
            } else if (EA_INSTANCE.weight_gap_length[biggest_patch_index] == 1) {
                sel_patch_idx = EA_INSTANCE.weight_gap_index[biggest_patch_index] - 1;
            } else {
                sel_patch_idx = EA_INSTANCE.weight_gap_index[biggest_patch_index]
                    - (EA_INSTANCE.weight_gap_length[biggest_patch_index] / 2) - 1;
            }
        #endif

        assert(sel_patch_idx >= 0);
        assert(sel_patch_idx < CMOCHC_PARETO_FRONT__PATCHES);

        #ifdef DEBUG_3
            fprintf(stderr, "[DEBUG] sel_patch_idx = %d\n", sel_patch_idx);
        #endif

        EA_INSTANCE.thread_weight_assignment[t] = sel_patch_idx;
        EA_INSTANCE.weight_thread_assignment[sel_patch_idx] = t;

        if (sel_patch_idx != EA_INSTANCE.weight_gap_index[biggest_patch_index]) {
            EA_INSTANCE.weight_gap_index[EA_INSTANCE.weight_gap_count] = sel_patch_idx;
            EA_INSTANCE.weight_gap_length[EA_INSTANCE.weight_gap_count] = EA_INSTANCE.weight_gap_length[biggest_patch_index] 
                - EA_INSTANCE.weight_gap_index[biggest_patch_index] + sel_patch_idx;
            EA_INSTANCE.weight_gap_sorted[EA_INSTANCE.weight_gap_count] = EA_INSTANCE.weight_gap_count;
                
            #ifdef DEBUG_3
                fprintf(stderr, "[DEBUG] weight_gap_length[last_gap_index=%d] = %d\n", EA_INSTANCE.weight_gap_count, EA_INSTANCE.weight_gap_length[EA_INSTANCE.weight_gap_count]);
                fprintf(stderr, "[DEBUG] weight_gap_length[biggest_patch_index=%d] = %d\n", biggest_patch_index, EA_INSTANCE.weight_gap_length[biggest_patch_index]);
                fprintf(stderr, "[DEBUG] weight_gap_index[biggest_patch_index=%d] = %d\n", biggest_patch_index, EA_INSTANCE.weight_gap_index[biggest_patch_index]);
            #endif
                
            assert(EA_INSTANCE.weight_gap_length[EA_INSTANCE.weight_gap_count] >= 0);
            
            EA_INSTANCE.weight_gap_count++;

            EA_INSTANCE.weight_gap_length[biggest_patch_index] = 
                EA_INSTANCE.weight_gap_index[biggest_patch_index] - sel_patch_idx - 1;
                
            assert(EA_INSTANCE.weight_gap_length[biggest_patch_index] >= 0);
        }

        for (int j = EA_INSTANCE.weight_gap_count - 2; j < EA_INSTANCE.weight_gap_count; j++) {
            int pos = j;
            int aux;
            
            while ((pos > 0) && (EA_INSTANCE.weight_gap_length[EA_INSTANCE.weight_gap_sorted[pos]] 
                <= EA_INSTANCE.weight_gap_length[EA_INSTANCE.weight_gap_sorted[pos-1]])) {
                    
                aux = EA_INSTANCE.weight_gap_sorted[pos-1];
                EA_INSTANCE.weight_gap_sorted[pos-1] = EA_INSTANCE.weight_gap_sorted[pos];
                EA_INSTANCE.weight_gap_sorted[pos] = aux;
                
                pos--;
            }
        }

        biggest_patch_index = EA_INSTANCE.weight_gap_sorted[EA_INSTANCE.weight_gap_count - 1];

        #ifdef DEBUG_3
            fprintf(stderr, "[DEBUG] Assigned thread %d. New gaps:\n", t);
            for (int i = 0; i < EA_INSTANCE.weight_gap_count; i++) {
                fprintf(stderr, "> [index=%d]<offset=%d> pos=%d size=%d\n", i, EA_INSTANCE.weight_gap_sorted[i],
                    EA_INSTANCE.weight_gap_index[EA_INSTANCE.weight_gap_sorted[i]], 
                    EA_INSTANCE.weight_gap_length[EA_INSTANCE.weight_gap_sorted[i]]);
            }
        #endif
    }
    
    #ifdef DEBUG_3
        fprintf(stderr, "[DEBUG] Thread assignments:\n");
        for (int t = 0; t < INPUT.thread_count; t++) {
            fprintf(stderr, "> thread %d=%d\n", t, EA_INSTANCE.thread_weight_assignment[t]);
        }
    #endif
    
    return 0;
}

/* Inicializa los hilos y las estructuras de datos */
void init(struct cmochc_thread *threads_data) {
    fprintf(stderr, "[INFO] == CMOCHC/islands configuration constants ==============\n");
    fprintf(stderr, "       CMOCHC_LOCAL__POPULATION_SIZE               : %d\n", CMOCHC_LOCAL__POPULATION_SIZE);
    fprintf(stderr, "       CMOCHC_LOCAL__ITERATION_COUNT               : %d\n", CMOCHC_LOCAL__ITERATION_COUNT);
    fprintf(stderr, "       CMOCHC_LOCAL__BEST_SOLS_KEPT                : %d\n", CMOCHC_LOCAL__BEST_SOLS_KEPT);
    fprintf(stderr, "       CMOCHC_LOCAL__MATING_MAX_THRESHOLD_DIVISOR  : %d\n", CMOCHC_LOCAL__MATING_MAX_THRESHOLD_DIVISOR);
    fprintf(stderr, "       CMOCHC_LOCAL__MATING_THRESHOLD_STEP_DIVISOR : %d\n", CMOCHC_LOCAL__MATING_THRESHOLD_STEP_DIVISOR);


    fprintf(stderr, "       CMOCHC_LOCAL__MUTATE_INITIAL_POP            : ");
    #ifdef CMOCHC_LOCAL__MUTATE_INITIAL_POP
        fprintf(stderr, "YES\n");
    #else
        fprintf(stderr, "NO\n");
    #endif

    fprintf(stderr, "       CMOCHC_LOCAL__FITNESS_NORM                  : ");
    #ifdef CMOCHC_LOCAL__Z_FITNESS_NORM
        fprintf(stderr, "Z (zenith)\n");
    #endif
    #ifdef CMOCHC_LOCAL__ZN_FITNESS_NORM
        fprintf(stderr, "ZN (zenith/nadir)\n");
    #endif

    fprintf(stderr, "       CMOCHC_PARETO_FRONT__PATCHES                : %d\n", CMOCHC_PARETO_FRONT__PATCHES);
    fprintf(stderr, "       CMOCHC_PARETO_FRONT__ADAPT_WEIGHTS          : ");    
    #ifdef CMOCHC_PARETO_FRONT__FIXED_WEIGHTS
        fprintf(stderr, "FIXED\n");
    #endif
    #ifdef CMOCHC_PARETO_FRONT__RANDOM_WEIGHTS
        fprintf(stderr, "RANDOM\n");
    #endif
    #ifdef CMOCHC_PARETO_FRONT__ADAPT_AR_WEIGHTS
        fprintf(stderr, "A-RANDOM\n");
    #endif
    #ifdef CMOCHC_PARETO_FRONT__ADAPT_AM_WEIGHTS
        fprintf(stderr, "A-MIDDLE\n");
    #endif
    
    fprintf(stderr, "       CMOCHC_ARCHIVE__MAX_SIZE                    : %d\n", CMOCHC_ARCHIVE__MAX_SIZE);   
    fprintf(stderr, "       CMOCHC_COLLABORATION__MOEAD_NEIGH_SIZE      : %d\n", CMOCHC_COLLABORATION__MOEAD_NEIGH_SIZE);
    fprintf(stderr, "       CMOCHC_COLLABORATION__MIGRATION             : ");
    #ifdef CMOCHC_COLLABORATION__MIGRATION_BEST
        fprintf(stderr, "BEST\n");
    #endif
    #ifdef CMOCHC_COLLABORATION__MIGRATION_RANDOM_ELITE
        fprintf(stderr, "RANDOM_ELITE\n");
    #endif
    #ifdef CMOCHC_COLLABORATION__MIGRATION_NONE
        fprintf(stderr, "NONE\n");
    #endif
    fprintf(stderr, "[INFO] ========================================================\n");

    /* Estado relacionado con el problema. */
    EA_INSTANCE.stopping_condition = 0;

    /* Weights */
    assert(CMOCHC_PARETO_FRONT__PATCHES > 1);

    for (int patch_idx = 0; patch_idx < CMOCHC_PARETO_FRONT__PATCHES; patch_idx++) {
        EA_INSTANCE.weights[patch_idx] = (FLOAT)(patch_idx+1) / (FLOAT)(CMOCHC_PARETO_FRONT__PATCHES+1);
        EA_INSTANCE.weight_thread_assignment[patch_idx] = -1;
        
        #ifdef DEBUG_2
            fprintf(stderr, "[DEBUG] weights[%d] = (%.4f, %.4f)\n", patch_idx, EA_INSTANCE.weights[patch_idx], 1 - EA_INSTANCE.weights[patch_idx]);
        #endif
    }
    
    if (INPUT.thread_count > 1) {
        for (int i = 0; i < INPUT.thread_count; i++) {
            EA_INSTANCE.thread_weight_assignment[i] = i * (CMOCHC_PARETO_FRONT__PATCHES / INPUT.thread_count);
            assert(EA_INSTANCE.thread_weight_assignment[i] < CMOCHC_PARETO_FRONT__PATCHES);
        
            EA_INSTANCE.weight_thread_assignment[EA_INSTANCE.thread_weight_assignment[i]] = i;
            
            #ifdef DEBUG_2
                fprintf(stderr, "[DEBUG] thread[%d] assigned to patch %d\n", i, EA_INSTANCE.thread_weight_assignment[i]);
            #endif
        }
    } else {
        EA_INSTANCE.thread_weight_assignment[0] = CMOCHC_PARETO_FRONT__PATCHES / 2;
        assert(EA_INSTANCE.thread_weight_assignment[0] < CMOCHC_PARETO_FRONT__PATCHES);

        EA_INSTANCE.weight_thread_assignment[EA_INSTANCE.thread_weight_assignment[0]] = 0;
        
        #ifdef DEBUG_2
            fprintf(stderr, "[DEBUG] thread[%d] assigned to patch %d\n", 0, EA_INSTANCE.thread_weight_assignment[0]);
        #endif
    }

    /* Sync */
    if (pthread_barrier_init(&(EA_INSTANCE.sync_barrier), NULL, INPUT.thread_count + 1))
    {
        fprintf(stderr, "[ERROR] could not create a sync barrier.\n");
        exit(EXIT_FAILURE);
    }

    /* Estado de la población elite global */
    for (int i = 0; i < (INPUT.thread_count * CMOCHC_LOCAL__BEST_SOLS_KEPT); i++) {
        create_empty_solution(&EA_INSTANCE.iter_elite_pop[i]);
    }

    /* Inicializo los hilos */
    for (int i = 0; i < INPUT.thread_count; i++)
    {
        struct cmochc_thread *t_data;
        t_data = &(threads_data[i]);
        t_data->thread_id = i;

        if (pthread_create(&(EA_INSTANCE.threads[i]), NULL, slave_thread, (void*) t_data))
        {
            fprintf(stderr, "[ERROR] could not create slave thread %d\n", i);
            exit(EXIT_FAILURE);
        }
    }

    /* Inicializo el archivador */
    archivers_aga_init(&EA_INSTANCE.archiver, CMOCHC_ARCHIVE__MAX_SIZE, EA_INSTANCE.iter_elite_pop, 
        EA_INSTANCE.iter_elite_pop_tag, (INPUT.thread_count * CMOCHC_LOCAL__BEST_SOLS_KEPT), 
        CMOCHC_PARETO_FRONT__PATCHES);
}

/* Obtiene los mejores elementos de cada población */
int gather() {
    #ifdef DEBUG_3
        fprintf(stderr, "[DEBUG] Gathering...\n");
        fprintf(stderr, "[DEBUG] Current iteration elite solutions:\n");

        int cantidad = 0;
        for (int i = 0; i < EA_INSTANCE.archiver.new_solutions_size; i++) {
            if (EA_INSTANCE.archiver.new_solutions[i].initialized == 1) cantidad++;

            fprintf(stderr, "> %d state=%d makespan=%f energy=%f\n",
                i, EA_INSTANCE.archiver.new_solutions[i].initialized,
                EA_INSTANCE.archiver.new_solutions[i].makespan,
                EA_INSTANCE.archiver.new_solutions[i].energy_consumption);
        }

        assert(cantidad > 0);
    #endif

    int new_solutions;
    new_solutions = archivers_aga(&EA_INSTANCE.archiver);

    #ifdef DEBUG_3
        fprintf(stderr, "[DEBUG] Total solutions gathered      = %d\n", new_solutions);
        fprintf(stderr, "[DEBUG] Current solutions in archiver = %d\n", EA_INSTANCE.archiver.population_count);
    #endif
    
    #ifdef DEBUG_1
        int current_tag;
        for (int s = 0; s < EA_INSTANCE.archiver.population_size; s++) {
            if (EA_INSTANCE.archiver.population[s].initialized == SOLUTION__IN_USE) {
                current_tag = EA_INSTANCE.archiver.population_tag[s];
                COUNT_HISTORIC_WEIGHTS[current_tag]++;
            }
        }
    #endif
    
    return new_solutions;
}

/* Logica de los esclavos */
void* slave_thread(void *data) {
    struct cmochc_thread *t_data = (struct cmochc_thread*)data;
    int thread_id = t_data->thread_id;

    /* *********************************************************************************************
     * Inicializo el thread.
     * *********************************************************************************************/

    /* Inicialización del estado del generador aleatorio */
    RAND_INIT(thread_id, EA_INSTANCE.rand_state[thread_id]);
    FLOAT random;

    /* Inicializo la población de padres y limpio la de hijos */
    int max_pop_sols = 2 * CMOCHC_LOCAL__POPULATION_SIZE;

    /* Merge sort tmp array */
    int merge_sort_tmp[2 * CMOCHC_LOCAL__POPULATION_SIZE];

    /* *********************************************************************************************
     * Inicializo los pesos.
     * *********************************************************************************************/

    int currently_assigned_weight = EA_INSTANCE.thread_weight_assignment[thread_id];
    FLOAT weight_makespan = EA_INSTANCE.weights[currently_assigned_weight];
    FLOAT energy_makespan = 1 - weight_makespan;

    /* *********************************************************************************************
     * Inicializo la población.
     * *********************************************************************************************/

    /* Poblacion de cada esclavo */
    struct solution population[2 * CMOCHC_LOCAL__POPULATION_SIZE];
    int sorted_population[2 * CMOCHC_LOCAL__POPULATION_SIZE];
    FLOAT fitness_population[2 * CMOCHC_LOCAL__POPULATION_SIZE];

    FLOAT makespan_utopia_value, energy_utopia_value;
    FLOAT makespan_nadir_value, energy_nadir_value;

    for (int i = 0; i < max_pop_sols; i++) {
        // Random init.
        create_empty_solution(&(population[i]));

        random = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]);
        int starting_pos;
        starting_pos = (int)(floor(INPUT.tasks_count * random));

        #ifdef DEBUG_3
            fprintf(stderr, "[DEBUG] Thread %d, inicializando solution %d, starting %d, direction %d...\n",
                thread_id, i, starting_pos, i & 0x1);
        #endif

        compute_mct_random(&(population[i]), starting_pos, i & 0x1);

        if (i == 0) {
            makespan_utopia_value = population[i].makespan;
            makespan_nadir_value = makespan_utopia_value;
            //makespan_utopia_index = i;

            energy_utopia_value = population[i].energy_consumption;
            energy_nadir_value = energy_utopia_value;
            //energy_utopia_index = i;
        } else {
            #ifdef CMOCHC_LOCAL__MUTATE_INITIAL_POP
                mutate(EA_INSTANCE.rand_state[thread_id], &population[i], &population[i]);
            #endif

            if (population[i].makespan < makespan_utopia_value) {
                //makespan_utopia_index = i;
                makespan_utopia_value = population[i].makespan;

                if (population[i].energy_consumption > energy_nadir_value) {
                    energy_nadir_value = population[i].energy_consumption;
                }
            }
            if (population[i].energy_consumption < energy_utopia_value) {
                //energy_utopia_index = i;
                energy_utopia_value = population[i].energy_consumption;

                if (population[i].makespan > makespan_nadir_value) {
                    makespan_nadir_value = population[i].makespan;
                }
            }
        }
    }

    #ifdef DEBUG_3
        fprintf(stderr, "[DEBUG] Normalization references\n");
        fprintf(stderr, "> (makespan) zenith=%.2f, nadir=%.2f\n",
            makespan_utopia_value, makespan_nadir_value);
        fprintf(stderr, "> (energy) zenith=%.2f, nadir=%.2f\n",
            energy_utopia_value, energy_nadir_value);
    #endif

    for (int i = 0; i < max_pop_sols; i++) {
        sorted_population[i] = i;
        fitness_population[i] = NAN;
        fitness(population, fitness_population, weight_makespan, energy_makespan,
            makespan_utopia_value, energy_utopia_value, makespan_nadir_value, energy_nadir_value, i);

        #ifdef DEBUG_3
            fprintf(stderr, "[DEBUG] Thread %d, solution=%d makespan=%.2f energy=%.2f fitness=%.2f\n",
                thread_id, i, population[i].makespan, population[i].energy_consumption,
                    fitness(population, fitness_population, weight_makespan, energy_makespan,
                        makespan_utopia_value, energy_utopia_value, makespan_nadir_value, 
                        energy_nadir_value, i));
        #endif
    }

    merge_sort(population, weight_makespan, energy_makespan, makespan_utopia_value, energy_utopia_value,
        makespan_nadir_value, energy_nadir_value, sorted_population, fitness_population, max_pop_sols,
        merge_sort_tmp);

    /* *********************************************************************************************
     * Main iteration
     * ********************************************************************************************* */
    int next_avail_children;
    int max_children = CMOCHC_LOCAL__POPULATION_SIZE / 2;
    int max_distance = INPUT.tasks_count;

    int threshold_max = max_distance / CMOCHC_LOCAL__MATING_MAX_THRESHOLD_DIVISOR;
    int threshold_step = threshold_max / CMOCHC_LOCAL__MATING_THRESHOLD_STEP_DIVISOR;
    if (threshold_step == 0) threshold_step = 1;
    int threshold = threshold_max;

    #ifdef DEBUG_1
        fprintf(stderr, "[DEBUG] Threshold Max %d.\n", threshold_max);
        fprintf(stderr, "[DEBUG] Threshold Step %d.\n", threshold_step);
    #endif

    int migration_global_pop_index[CMOCHC_COLLABORATION__MOEAD_NEIGH_SIZE];
    int migration_current_weight_distance[CMOCHC_COLLABORATION__MOEAD_NEIGH_SIZE];
    
    int rc;

    while (EA_INSTANCE.stopping_condition == 0) {
        for (int iteracion = 0; iteracion < CMOCHC_LOCAL__ITERATION_COUNT; iteracion++) {
            #ifdef DEBUG_1
                COUNT_GENERATIONS[thread_id]++;
            #endif

            /* *********************************************************************************************
             * Mating
             * ********************************************************************************************* */
            next_avail_children = CMOCHC_LOCAL__POPULATION_SIZE;

            FLOAT d;
            int p1_idx, p2_idx;
            int p1_rand, p2_rand;
            int c1_idx, c2_idx;
            for (int child = 0; child < max_children; child++) {
                if (next_avail_children + 1 < max_pop_sols) {
                    // Padre aleatorio 1
                    random = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]);
                    p1_rand = (int)(floor(CMOCHC_LOCAL__POPULATION_SIZE * random));
                    p1_idx = sorted_population[p1_rand];

                    // Padre aleatorio 2
                    random = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]);
                    p2_rand = (int)(floor((CMOCHC_LOCAL__POPULATION_SIZE - 1) * random));
                    if (p2_rand >= p1_rand) p2_rand++;
                    p2_idx = sorted_population[p2_rand];

                    // Chequeo la distancia entre padres
                    d = distance(&population[p1_idx],&population[p2_idx]);

                    if (d > threshold) {
                        // Aplico HUX y creo dos hijos
                        COUNT_CROSSOVER[thread_id]++;

                        c1_idx = sorted_population[next_avail_children];
                        c2_idx = sorted_population[next_avail_children+1];

                        hux(EA_INSTANCE.rand_state[thread_id],
                            &population[p1_idx],&population[p2_idx],
                            &population[c1_idx],&population[c2_idx]);

                        fitness_population[c1_idx] = NAN;
                        fitness_population[c2_idx] = NAN;

                        fitness(population, fitness_population, weight_makespan, energy_makespan,
                            makespan_utopia_value, energy_utopia_value, makespan_nadir_value, energy_nadir_value, c1_idx);
                        fitness(population, fitness_population, weight_makespan, energy_makespan,
                            makespan_utopia_value, energy_utopia_value, makespan_nadir_value, energy_nadir_value, c2_idx);

                        #ifdef DEBUG_1
                            if ((fitness_population[c1_idx] < fitness_population[p1_idx])
                                ||(fitness_population[c1_idx] < fitness_population[p2_idx])
                                ||(fitness_population[c2_idx] < fitness_population[p1_idx])
                                ||(fitness_population[c2_idx] < fitness_population[p2_idx])) {

                                COUNT_IMPROVED_CROSSOVER[thread_id]++;
                            }
                        #endif

                        next_avail_children += 2;
                    }
                }
            }

            if (next_avail_children > CMOCHC_LOCAL__POPULATION_SIZE) {
                /* *********************************************************************************************
                 * Sort parent+children population
                 * ********************************************************************************************* */

                FLOAT best_parent;
                best_parent = fitness(population, fitness_population, weight_makespan, energy_makespan,
                    makespan_utopia_value, energy_utopia_value, makespan_nadir_value,
                    energy_nadir_value, sorted_population[0]);

                FLOAT worst_parent;
                worst_parent = fitness(population, fitness_population, weight_makespan, energy_makespan,
                    makespan_utopia_value, energy_utopia_value, makespan_nadir_value,
                    energy_nadir_value, sorted_population[CMOCHC_LOCAL__POPULATION_SIZE-1]);

                merge_sort(population, weight_makespan, energy_makespan, makespan_utopia_value, energy_utopia_value,
                    makespan_nadir_value, energy_nadir_value, sorted_population,
                    fitness_population, max_pop_sols, merge_sort_tmp);

                if (worst_parent > fitness(population, fitness_population, weight_makespan, energy_makespan,
                        makespan_utopia_value, energy_utopia_value, makespan_nadir_value,
                        energy_nadir_value, sorted_population[CMOCHC_LOCAL__POPULATION_SIZE-1])) {

                    #ifdef DEBUG_1
                        COUNT_AT_LEAST_ONE_CHILDREN_INSERTED[thread_id]++;
                    #endif
                } else {
                    threshold -= threshold_step;
                }

                if (best_parent > fitness(population, fitness_population, weight_makespan, energy_makespan,
                        makespan_utopia_value, energy_utopia_value, makespan_nadir_value,
                        energy_nadir_value, sorted_population[0])) {

                    #ifdef DEBUG_1
                        COUNT_IMPROVED_BEST_SOL[thread_id]++;
                    #endif
                }
            } else {
                threshold -= threshold_step;
            }

            if (threshold < 0) {
                threshold = threshold_max;

                /* *********************************************************************************************
                 * Cataclysm
                 * ********************************************************************************************* */

                #ifdef DEBUG_1
                    FLOAT pre_mut_fitness;
                #endif

                #ifdef DEBUG_3
                    fprintf(stderr, "[DEBUG] Cataclysm (thread=%d)!\n", thread_id);
                #endif

                int aux_index;

                for (int i = CMOCHC_LOCAL__BEST_SOLS_KEPT; i < max_pop_sols; i++) { /* No muto las mejores soluciones */
                    if (population[sorted_population[i]].initialized == SOLUTION__IN_USE) {
                        #ifdef DEBUG_1
                            COUNT_CATACLYSM[thread_id]++;
                            pre_mut_fitness = fitness(population, fitness_population, weight_makespan, energy_makespan, 
                                makespan_utopia_value, energy_utopia_value, makespan_nadir_value, energy_nadir_value, 
                                sorted_population[i]);
                        #endif

                        aux_index = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]) * CMOCHC_LOCAL__BEST_SOLS_KEPT;

                        mutate(EA_INSTANCE.rand_state[thread_id],
                            &population[sorted_population[aux_index]],
                            &population[sorted_population[i]]);

                        fitness_population[sorted_population[i]] = NAN;
                        fitness(population, fitness_population, weight_makespan, energy_makespan, makespan_utopia_value, 
                            energy_utopia_value, makespan_nadir_value, energy_nadir_value, sorted_population[i]);

                        #ifdef DEBUG_1
                            if (fitness_population[sorted_population[i]] < pre_mut_fitness) {
                                COUNT_IMPOVED_CATACLYSM[thread_id]++;
                            }
                        #endif
                    }
                }

                /* Re-sort de population */
                merge_sort(population, weight_makespan, energy_makespan, makespan_utopia_value, energy_utopia_value,
                    makespan_nadir_value, energy_nadir_value, sorted_population,
                    fitness_population, max_pop_sols, merge_sort_tmp);
            }
        }

        /* *********************************************************************************************
         * Fin de iteracion local
         * ********************************************************************************************* */

        /* Copio mis mejores soluciones a la población de intercambio principal */
        int iter_elite_pop_index;
        int local_best_index;
      
        for (int i = 0; i < CMOCHC_LOCAL__BEST_SOLS_KEPT; i++) {
            iter_elite_pop_index = thread_id * CMOCHC_LOCAL__BEST_SOLS_KEPT + i;
            local_best_index = sorted_population[i];
        
            #ifdef DEBUG_3
                fprintf(stderr, "[DEBUG] Thread %d, copying from %d to %d\n", 
                    thread_id, local_best_index, iter_elite_pop_index);
            #endif
        
            clone_solution(&EA_INSTANCE.iter_elite_pop[iter_elite_pop_index], &population[local_best_index]);
            EA_INSTANCE.iter_elite_pop_tag[iter_elite_pop_index] = EA_INSTANCE.thread_weight_assignment[thread_id];
        }

        /* Le aviso al maestro que puede empezar con la operación de gather. */
        rc = pthread_barrier_wait(&EA_INSTANCE.sync_barrier);
        if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
        {
            printf("Could not wait on barrier\n");
            exit(EXIT_FAILURE);
        }

        // ..................

        /* Espero a que el maestro ejecute la operación de gather. */
        rc = pthread_barrier_wait(&EA_INSTANCE.sync_barrier);
        if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
        {
            printf("Could not wait on barrier\n");
            exit(EXIT_FAILURE);
        }

        if (EA_INSTANCE.stopping_condition == 0) {
            /* Actualizo la nueva asignación de pesos y vecindad */
            #ifdef DEBUG_3
                fprintf(stderr, "[DEBUG] Thread %d, Current weights=%d (%.4f,%.4f), new weights=%d (%.4f,%.4f)\n",
                    thread_id, currently_assigned_weight, weight_makespan, energy_makespan, 
                    EA_INSTANCE.thread_weight_assignment[thread_id], 
                    EA_INSTANCE.weights[EA_INSTANCE.thread_weight_assignment[thread_id]], 
                    1 - EA_INSTANCE.weights[EA_INSTANCE.thread_weight_assignment[thread_id]]);
            #endif
            
            //int changed_assignment;
            
            if (currently_assigned_weight != EA_INSTANCE.thread_weight_assignment[thread_id]) {
                //changed_assignment = 1;
                
                weight_makespan = EA_INSTANCE.weights[EA_INSTANCE.thread_weight_assignment[thread_id]];
                energy_makespan = 1 - weight_makespan;
            } else {
                //changed_assignment = 0;
            }

            /* *********************************************************************************************
             * Busco las mejores soluciones elite a importar
             * ********************************************************************************************* */
            #ifndef CMOCHC_COLLABORATION__MIGRATION_NONE
                for (int i = 0; i < CMOCHC_COLLABORATION__MOEAD_NEIGH_SIZE; i++) {
                    migration_global_pop_index[i] = -1;
                }
                
                int ep_current_index, ep_solution_index;
                ep_current_index = 0;
                ep_solution_index = 0;

                #ifdef CMOCHC_COLLABORATION__MIGRATION_BEST
                    int worst_distance, worst_index;
                    int current_solution_distance;
                            
                    while ((ep_current_index < EA_INSTANCE.archiver.population_size) &&
                        (ep_solution_index < EA_INSTANCE.archiver.population_count)) {
                            
                        if (EA_INSTANCE.archiver.population[ep_current_index].initialized == SOLUTION__IN_USE) {
                            if (ep_solution_index < CMOCHC_COLLABORATION__MOEAD_NEIGH_SIZE) {
                                /* Aún no esta lleno el array con el vecindario */
                                migration_global_pop_index[ep_solution_index] = ep_current_index;
                                migration_current_weight_distance[ep_solution_index] = abs(EA_INSTANCE.thread_weight_assignment[thread_id] -
                                    EA_INSTANCE.archiver.population_tag[ep_current_index]);
                                    
                                if (ep_solution_index == 0) {
                                    worst_distance = migration_current_weight_distance[ep_solution_index];
                                    worst_index = ep_solution_index;
                                } else {
                                    if (worst_distance < migration_current_weight_distance[ep_solution_index]) {
                                        worst_distance = migration_current_weight_distance[ep_solution_index];
                                        worst_index = ep_solution_index;
                                    }
                                }
                            } else {
                                current_solution_distance = abs(EA_INSTANCE.thread_weight_assignment[thread_id] -
                                    EA_INSTANCE.archiver.population_tag[ep_current_index]);
                                    
                                if (current_solution_distance < worst_distance) {
                                    worst_distance = current_solution_distance;
                                    
                                    migration_global_pop_index[worst_index] = ep_current_index;
                                    migration_current_weight_distance[worst_index] = current_solution_distance;
                                    
                                    for (int i = 0; i < CMOCHC_COLLABORATION__MOEAD_NEIGH_SIZE; i++) {
                                        if (worst_distance < migration_current_weight_distance[i]) {
                                            worst_distance = migration_current_weight_distance[i];
                                            worst_index = i;
                                        }
                                    }
                                }
                            }
                            
                            ep_solution_index++;
                        }
                        
                        ep_current_index++;
                    }
                #endif
                #ifdef CMOCHC_COLLABORATION__MIGRATION_RANDOM_ELITE
                    int current_ep_solution_index;
                    current_ep_solution_index = 0;
                    
                    FLOAT selection_prob;
                    selection_prob = 1 / EA_INSTANCE.archiver.population_count;
                
                    while ((ep_current_index < EA_INSTANCE.archiver.population_size) &&
                        (ep_solution_index < EA_INSTANCE.archiver.population_count) &&
                        (current_ep_solution_index < CMOCHC_COLLABORATION__MOEAD_NEIGH_SIZE)) {
                            
                        if (EA_INSTANCE.archiver.population[ep_current_index].initialized == SOLUTION__IN_USE) {
                            if (RAND_GENERATE(rand_state[thread_id]) <= selection_prob) {
                                /* Aún no esta lleno el array con el vecindario */
                                migration_global_pop_index[current_ep_solution_index] = ep_current_index;
                                migration_current_weight_distance[current_ep_solution_index] = 
                                    abs(thread_weight_assignment[thread_id] -
                                        EA_INSTANCE.archiver.population_tag[ep_current_index]);
                                
                                current_ep_solution_index++;
                            }
                            
                            ep_solution_index++;
                        }
                        
                        ep_current_index++;
                    }
                #endif
                
                #ifdef DEBUG_3
                    for (int i = 0; i < CMOCHC_COLLABORATION__MOEAD_NEIGH_SIZE; i++) {
                        if (migration_global_pop_index[i] != -1) {                           
                            fprintf(stderr, "[DEBUG] Thread %d, Neighbour %d (weight idx %d, distance %d).\n", 
                                thread_id, 
                                migration_global_pop_index[i], 
                                EA_INSTANCE.archiver.population_tag[migration_global_pop_index[i]],
                                migration_current_weight_distance[i]);
                        }
                    } 
                #endif
            #endif

            /* *********************************************************************************************
             * Migro soluciones desde la población elite
             * ********************************************************************************************* */
             
            int migrated;
            
            int migrated_solution_index;
            migrated_solution_index = 0;
            
            int next_solution_index;
            next_solution_index = 0;
            
            while (next_solution_index < max_pop_sols) {
                migrated = 0;
                
                #ifndef CMOCHC_COLLABORATION__MIGRATION_NONE
                    if (migrated_solution_index < CMOCHC_COLLABORATION__MOEAD_NEIGH_SIZE) {
                        if (migration_global_pop_index[migrated_solution_index] != -1) {
                            COUNT_MIGRATIONS[thread_id]++;
                        
                            #ifdef CMOCHC_COLLABORATION__MIGRATE_BY_COPY
                                clone_solution(&population[next_solution_index],
                                    &EA_INSTANCE.archiver.population[migration_global_pop_index[migrated_solution_index]]);
                                    
                                next_solution_index++;
                                migrated_solution_index++;
                                count_solutions_migrated[thread_id]++;
                                migrated = 1;
                            #endif
                            #ifdef CMOCHC_COLLABORATION__MIGRATE_BY_MATE
                                if (next_solution_index + 1 < max_pop_sols) {
                                    random = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]);
                                    
                                    int migration_parent_index;
                                    migration_parent_index = (int)(random * next_solution_index);
                                    
                                    FLOAT d = distance(
                                        &EA_INSTANCE.archiver.population[migration_global_pop_index[migrated_solution_index]],
                                        &population[migration_parent_index]);

                                    if (d > threshold_max) {
                                        hux(EA_INSTANCE.rand_state[thread_id],
                                            &EA_INSTANCE.archiver.population[migration_global_pop_index[migrated_solution_index]], 
                                            &population[migration_parent_index],
                                            &population[sorted_population[next_solution_index]],
                                            &population[sorted_population[next_solution_index+1]]);
                                            
                                        next_solution_index += 2;
                                        migrated_solution_index++;
                                        COUNT_SOLUTIONS_MIGRATED[thread_id] += 2;
                                        migrated = 1;
                                    }
                                }
                                
                                if (migrated == 0) {
                                    mutate(
                                        EA_INSTANCE.rand_state[thread_id],
                                        &EA_INSTANCE.archiver.population[migration_global_pop_index[migrated_solution_index]],
                                        &population[next_solution_index]);
                                        
                                    next_solution_index++;
                                    migrated_solution_index++;
                                    COUNT_SOLUTIONS_MIGRATED[thread_id]++; 
                                    migrated = 1;
                                }
                            #endif
                            #ifdef CMOCHC_COLLABORATION__MIGRATE_BY_MUTATE
                                mutate(
                                    EA_INSTANCE.rand_state[thread_id],
                                    &EA_INSTANCE.archiver.population[migration_global_pop_index[migrated_solution_index]],
                                    &population[next_solution_index]);
                                    
                                next_solution_index++;
                                migrated_solution_index++;
                                COUNT_SOLUTIONS_MIGRATED[thread_id]++;
                                
                                migrated = 1;
                            #endif
                        }
                    }
                #endif
                
                if (migrated == 0) {
                    random = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]);

                    mutate(EA_INSTANCE.rand_state[thread_id],
                        &population[(int)(random * next_solution_index)],
                        &population[next_solution_index]);

                    next_solution_index++;
                }
            }
           
            /* Actualizo los puntos de normalización con la población local */
            for (int i = 0; i < max_pop_sols; i++) {
                if (population[i].makespan < makespan_utopia_value) {
                    //makespan_utopia_index = i;
                    makespan_utopia_value = population[i].makespan;

                    if (population[i].energy_consumption > energy_nadir_value) {
                        energy_nadir_value = population[i].energy_consumption;
                    }
                }
                if (population[i].energy_consumption < energy_utopia_value) {
                    //energy_utopia_index = i;
                    energy_utopia_value = population[i].energy_consumption;

                    if (population[i].makespan > makespan_nadir_value) {
                        makespan_nadir_value = population[i].makespan;
                    }
                }
            }

            /* *********************************************************************************************
             * Re-calculo el fitness de toda la población
             *********************************************************************************************** */

            for (int i = 0; i < max_pop_sols; i++) {
                fitness_population[i] = NAN;
                fitness(population, fitness_population, weight_makespan, energy_makespan, makespan_utopia_value, 
                    energy_utopia_value, makespan_nadir_value, energy_nadir_value, i);
            }

            /* *********************************************************************************************
             * Re-sort de population
             *********************************************************************************************** */
            merge_sort(population, weight_makespan, energy_makespan, makespan_utopia_value, energy_utopia_value,
                makespan_nadir_value, energy_nadir_value, sorted_population,
                fitness_population, max_pop_sols, merge_sort_tmp);

            #ifdef DEBUG_3
                fprintf(stderr, "[DEBUG] Post-migration population\n");
                fprintf(stderr, "parents> ");
                for (int i = 0; i < CMOCHC_LOCAL__POPULATION_SIZE; i++) {
                    fprintf(stderr, "%d(%f)<%d>  ", sorted_population[i], 
                        fitness_population[sorted_population[i]], 
                        population[sorted_population[i]].initialized);
                }
                fprintf(stderr, "\n");
                fprintf(stderr, "childs > ");
                for (int i = CMOCHC_LOCAL__POPULATION_SIZE; i < max_pop_sols; i++) {
                    fprintf(stderr, "%d(%f)<%d>  ", sorted_population[i], 
                        fitness_population[sorted_population[i]], 
                        population[sorted_population[i]].initialized);
                }
                fprintf(stderr, "\n");
            #endif
        }
    }

    // ================================================================
    // Finalizo el thread.
    // ================================================================   

    RAND_FINALIZE(EA_INSTANCE.rand_state[thread_id]);
    
    for (int p = 0; p < max_pop_sols; p++) {
        free_solution(&population[p]);
    }

    return 0;
}

void display_results() {
    /* Show solutions */
    #if defined(OUTPUT_SOLUTION)
        archivers_aga_dump(&EA_INSTANCE.archiver);
    #endif

    #ifdef DEBUG_1
        archivers_aga_show(&EA_INSTANCE.archiver);

        int count_generations = 0;
        int count_at_least_one_children_inserted = 0;
        int count_improved_best_sol = 0;
        int count_crossover = 0;
        int count_improved_crossover = 0;
        int count_cataclysm = 0;
        int count_improved_mutation = 0;
        int count_migrations = 0;
        int count_solutions_migrated = 0;

        for (int t = 0; t < INPUT.thread_count; t++) {
            count_generations += COUNT_GENERATIONS[t];
            count_at_least_one_children_inserted += COUNT_AT_LEAST_ONE_CHILDREN_INSERTED[t];
            count_improved_best_sol += COUNT_IMPROVED_BEST_SOL[t];
            count_crossover += COUNT_CROSSOVER[t];
            count_improved_crossover += COUNT_IMPROVED_CROSSOVER[t];
            count_cataclysm += COUNT_CATACLYSM[t];
            count_improved_mutation += COUNT_IMPOVED_CATACLYSM[t];
            count_migrations += COUNT_MIGRATIONS[t];
            count_solutions_migrated += COUNT_SOLUTIONS_MIGRATED[t];
        }

        int *count_pf_found;
        count_pf_found = (int*)(malloc(sizeof(int) * CMOCHC_PARETO_FRONT__PATCHES));

        for (int s = 0; s < CMOCHC_PARETO_FRONT__PATCHES; s++) {
            count_pf_found[s] = 0;
        }

        for (int s = 0; s < EA_INSTANCE.archiver.population_size; s++) {
            if (EA_INSTANCE.archiver.population[s].initialized == SOLUTION__IN_USE) {
                count_pf_found[EA_INSTANCE.archiver.population_tag[s]]++;
            }
        }

        fprintf(stderr, "[INFO] == Statistics ==========================================\n");
        fprintf(stderr, "       count_generations                    : %d\n", count_generations);
        fprintf(stderr, "       count_at_least_one_children_inserted : %d (%.2f %%)\n", count_at_least_one_children_inserted,
            ((FLOAT)count_at_least_one_children_inserted/(FLOAT)count_generations)*100);
        fprintf(stderr, "       count_improved_best_sol              : %d (%.2f %%)\n", count_improved_best_sol,
            ((FLOAT)count_improved_best_sol/(FLOAT)count_generations)*100);
        fprintf(stderr, "       count_crossover                      : %d\n", count_crossover);
        fprintf(stderr, "       count_improved_crossover             : %d (%.2f %%)\n", count_improved_crossover,
            ((FLOAT)count_improved_crossover/(FLOAT)count_crossover)*100);
        fprintf(stderr, "       count_cataclysm                      : %d\n", count_cataclysm);
        fprintf(stderr, "       count_improved_mutation              : %d (%.2f %%)\n", count_improved_mutation,
            ((FLOAT)count_improved_mutation/(FLOAT)count_cataclysm)*100);
        fprintf(stderr, "       count_migrations                     : %d\n", count_migrations);
        fprintf(stderr, "       count_solutions_migrated             : %d (%.2f %%)\n", count_solutions_migrated,
            ((FLOAT)count_solutions_migrated/(FLOAT)count_migrations)*100);

        fprintf(stderr, "       archive tag count:\n");
        for (int t = 0; t < CMOCHC_PARETO_FRONT__PATCHES; t++) {
            fprintf(stderr, "          [%d] = %d\n", t, EA_INSTANCE.archiver.tag_count[t]);
        }

        fprintf(stderr, "       historic archive tag count:\n");
        for (int t = 0; t < CMOCHC_PARETO_FRONT__PATCHES; t++) {
            fprintf(stderr, "          [%d] = %d (%d)\n", t, count_pf_found[t], COUNT_HISTORIC_WEIGHTS[t]);
        }

        fprintf(stderr, "[INFO] ========================================================\n");
        
        free(count_pf_found);
    #endif
}

/* Libera los recursos pedidos y finaliza la ejecución */
void finalize() {
    archivers_aga_free(&EA_INSTANCE.archiver);
    pthread_barrier_destroy(&(EA_INSTANCE.sync_barrier));
        
    for (int i = 0; i < (INPUT.thread_count * CMOCHC_LOCAL__BEST_SOLS_KEPT); i++) {
        free_solution(&EA_INSTANCE.iter_elite_pop[i]);
    }
}
