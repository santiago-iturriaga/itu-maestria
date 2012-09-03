#include <pthread.h>
#include <math.h>
#include <stdlib.h>

#include "cmochc_island.h"

#include "../config.h"
#include "../solution.h"
#include "../load_params.h"
#include "../scenario.h"
#include "../etc_matrix.h"
#include "../energy_matrix.h"
#include "../utils.h"
#include "../basic/mct.h"
#include "../random/random.h"
#include "../archivers/aga.h"

#include "cmochc_island_utils.h"
#include "cmochc_island_evop.h"

struct cmochc {
    struct params *input;
    struct scenario *current_scenario;
    struct etc_matrix *etc;
    struct energy_matrix *energy;

    /* Coleccion de esclavos */
    pthread_t *threads;

    /* Poblacion elite global mantenida por el master */
    struct solution *iter_elite_pop;
    int *iter_elite_pop_tag;
    
    struct aga_state archiver;

    /* Descomposición del frente de pareto */
    FLOAT *weights;
    int *thread_weight_assignment;
    int *weight_thread_assignment;
    
    int stopping_condition;

    /* Random generator de cada esclavo y para el master */
    RAND_STATE *rand_state;

    /* Sync */
    pthread_barrier_t sync_barrier;

    /* Statistics */
    #ifdef DEBUG_1
        int *count_generations;
        int *count_at_least_one_children_inserted;
        int *count_improved_best_sol;
        int *count_crossover;
        int *count_improved_crossover;
        int *count_cataclysm;
        int *count_improved_mutation;
        int *count_migrations;
        int *count_solutions_migrated;
        int *count_historic_weights;
    #endif
};

struct cmochc_thread {
    /* Id del esclavo */
    int thread_id;

    struct cmochc *data;
};

/* Inicializa los hilos y las estructuras de datos */
void init(struct cmochc &instance, struct cmochc_thread **threads_data,
    struct params &input, struct scenario &current_scenario,
    struct etc_matrix &etc, struct energy_matrix &energy);

/* Obtiene los mejores elementos de cada población */
int gather(struct cmochc &instance);

/* Adapta los pesos de los threads */
int adapt_weights_mod_a(struct cmochc &instance);

/* Muestra el resultado de la ejecución */
void display_results(struct cmochc &instance);

/* Libera los recursos pedidos y finaliza la ejecución */
void finalize(struct cmochc &instance, struct cmochc_thread *threads);

/* Logica de los esclavos */
void* slave_thread(void *data);

void compute_cmochc_island(struct params &input, struct scenario &current_scenario,
    struct etc_matrix &etc, struct energy_matrix &energy) {

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

    struct cmochc instance;
    struct cmochc_thread *threads;
    init(instance, &threads, input, current_scenario, etc, energy);

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
    RAND_INIT(input.thread_count, rstate);
       
    for (int iteracion = 0; iteracion < input.max_iterations; iteracion++) {
        /* ************************************************ */
        /* Espero que los esclavos terminen de evolucionar  */
        /* ************************************************ */
        rc = pthread_barrier_wait(&instance.sync_barrier);
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

        sols_gathered = gather(instance);
        if (sols_gathered > 0) last_iter_sols_gathered = iteracion;

        #ifndef CMOCHC_PARETO_FRONT__FIXED_WEIGHTS
            #ifdef CMOCHC_PARETO_FRONT__RANDOM_WEIGHTS
                FLOAT random;
            
                for (int i = 0; i < input.thread_count; i++) {
                    random = RAND_GENERATE(rstate);
                    instance.thread_weight_assignment[i] = (int)(random * CMOCHC_PARETO_FRONT__PATCHES);
                }
            #endif
            #if defined(CMOCHC_PARETO_FRONT__ADAPT_AR_WEIGHTS) || defined(CMOCHC_PARETO_FRONT__ADAPT_AM_WEIGHTS)
                adapt_weights_mod_a(instance);
            #endif
        #endif

        TIMMING_END(">> cmochc_gather", ts_gather);

        if (iteracion + 1 >= input.max_iterations) {
            /* Si esta es la úlitma iteracion, les aviso a los esclavos */
            instance.stopping_condition = 1;
        }

        /* *********************************************************** */
        /* Notifico a los esclavos que terminó la operación de gather  */
        /* *********************************************************** */
        rc = pthread_barrier_wait(&instance.sync_barrier);
        if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
        {
            printf("Could not wait on barrier\n");
            exit(EXIT_FAILURE);
        }
    }

    RAND_FINALIZE(rstate);

    /* Bloqueo la ejecucion hasta que terminen todos los hilos. */
    for(int i = 0; i < instance.input->thread_count; i++)
    {
        if(pthread_join(instance.threads[i], NULL))
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

    display_results(instance);
    
    #if defined(DEBUG_1)
        fprintf(stderr, "[DEBUG] Last solution gathered on iteration = %d\n", last_iter_sols_gathered);
    #endif
    
    finalize(instance, threads);
}

int adapt_weights_mod_a(struct cmochc &instance) {
    int *weight_gap_sorted = (int*) malloc(sizeof(int) * (instance.archiver.population_size + instance.input->thread_count + 1));
    int *weight_gap_length = (int*) malloc(sizeof(int) * (instance.archiver.population_size + instance.input->thread_count + 1));
    int *weight_gap_index = (int*) malloc(sizeof(int) * (instance.archiver.population_size + instance.input->thread_count + 1));
    int *weight_gap_tmp = (int*) malloc(sizeof(int) * (instance.archiver.population_size + instance.input->thread_count + 1));
    
    int last_gap_index = 0;
    int last_filled_patch = -1;
    
    #ifdef DEBUG_3
        fprintf(stderr, "[DEBUG] Archive tag count:\n");
        for (int t = 0; t < CMOCHC_PARETO_FRONT__PATCHES; t++) {
            fprintf(stderr, "> [%d] = %d\n", t, instance.archiver.tag_count[t]);
        }
    #endif
    
    for (int i = 0; i < CMOCHC_PARETO_FRONT__PATCHES; i++) {
        if (instance.archiver.tag_count[i] > 0) {
            if (i > last_filled_patch + 1) {
                weight_gap_index[last_gap_index] = i;
                weight_gap_length[last_gap_index] = i - last_filled_patch - 1;
                weight_gap_sorted[last_gap_index] = last_gap_index;
                last_gap_index++;
            }
            
            last_filled_patch = i;
        }
    }
    
    if (CMOCHC_PARETO_FRONT__PATCHES > last_filled_patch + 1) {
        weight_gap_index[last_gap_index] = CMOCHC_PARETO_FRONT__PATCHES;
        weight_gap_length[last_gap_index] = CMOCHC_PARETO_FRONT__PATCHES - last_filled_patch - 1;
        weight_gap_sorted[last_gap_index] = last_gap_index;
        last_gap_index++;
    }
    
    #ifdef DEBUG_3
        fprintf(stderr, "[DEBUG] Found gaps:\n");
        for (int i = 0; i < last_gap_index; i++) {
            fprintf(stderr, "> [index=%d] pos=%d size=%d\n", i, weight_gap_index[i], weight_gap_length[i]);
        }
    #endif
    
    gap_merge_sort(weight_gap_index, weight_gap_length, weight_gap_sorted, last_gap_index, weight_gap_tmp);

    #ifdef DEBUG_3
        fprintf(stderr, "[DEBUG] Sorted found gaps:\n");
        for (int i = 0; i < last_gap_index; i++) {
            fprintf(stderr, "> [index=%d] pos=%d size=%d\n", i, weight_gap_index[weight_gap_sorted[i]], weight_gap_length[weight_gap_sorted[i]]);
        }
    #endif
    
    
    
    free(weight_gap_sorted);
    free(weight_gap_tmp);
    free(weight_gap_length);
    free(weight_gap_index);
}

/* Inicializa los hilos y las estructuras de datos */
void init(struct cmochc &instance, struct cmochc_thread **threads_data,
    struct params &input, struct scenario &current_scenario,
    struct etc_matrix &etc, struct energy_matrix &energy) {

    fprintf(stderr, "[INFO] == CMOCHC/islands configuration constants ==============\n");
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
    instance.input = &input;
    instance.current_scenario = &current_scenario;
    instance.etc = &etc;
    instance.energy = &energy;
    instance.stopping_condition = 0;

    /* Estado del generador aleatorio. */
    instance.rand_state = (RAND_STATE*)(malloc(sizeof(RAND_STATE) * input.thread_count));

    /* Weights */
    instance.weight_thread_assignment = (int*)(malloc(sizeof(int) * CMOCHC_PARETO_FRONT__PATCHES));
    instance.weights = (FLOAT*)(malloc(sizeof(FLOAT) * CMOCHC_PARETO_FRONT__PATCHES));
    ASSERT(CMOCHC_PARETO_FRONT__PATCHES > 1)

    for (int patch_idx = 0; patch_idx < CMOCHC_PARETO_FRONT__PATCHES; patch_idx++) {
        instance.weights[patch_idx] = (FLOAT)(patch_idx+1) / (FLOAT)(CMOCHC_PARETO_FRONT__PATCHES+1);
        instance.weight_thread_assignment[patch_idx] = -1;
        
        #ifdef DEBUG_2
            fprintf(stderr, "[DEBUG] weights[%d] = (%.4f, %.4f)\n", patch_idx, instance.weights[patch_idx], 1 - instance.weights[patch_idx]);
        #endif
    }
    
    instance.thread_weight_assignment = (int*)(malloc(sizeof(int) * input.thread_count));
    if (input.thread_count > 1) {
        for (int i = 0; i < input.thread_count; i++) {
            instance.thread_weight_assignment[i] = i * (CMOCHC_PARETO_FRONT__PATCHES / input.thread_count);
            ASSERT(instance.thread_weight_assignment[i] < CMOCHC_PARETO_FRONT__PATCHES)
        
            instance.weight_thread_assignment[instance.thread_weight_assignment[i]] = i;
            
            #ifdef DEBUG_2
                fprintf(stderr, "[DEBUG] thread[%d] assigned to patch %d\n", i, instance.thread_weight_assignment[i]);
            #endif
        }
    } else {
        instance.thread_weight_assignment[0] = CMOCHC_PARETO_FRONT__PATCHES / 2;
        ASSERT(instance.thread_weight_assignment[0] < CMOCHC_PARETO_FRONT__PATCHES)

        instance.weight_thread_assignment[instance.thread_weight_assignment[0]] = 0;
        
        #ifdef DEBUG_2
            fprintf(stderr, "[DEBUG] thread[%d] assigned to patch %d\n", 0, instance.thread_weight_assignment[0]);
        #endif
    }

    /* Statistics */
    #ifdef DEBUG_1
        instance.count_generations = (int*)(malloc(sizeof(int) * input.thread_count));
        instance.count_at_least_one_children_inserted = (int*)(malloc(sizeof(int) * input.thread_count));
        instance.count_improved_best_sol = (int*)(malloc(sizeof(int) * input.thread_count));
        instance.count_crossover = (int*)(malloc(sizeof(int) * input.thread_count));
        instance.count_improved_crossover = (int*)(malloc(sizeof(int) * input.thread_count));
        instance.count_cataclysm = (int*)(malloc(sizeof(int) * input.thread_count));
        instance.count_improved_mutation = (int*)(malloc(sizeof(int) * input.thread_count));
        instance.count_migrations = (int*)(malloc(sizeof(int) * input.thread_count));
        instance.count_solutions_migrated = (int*)(malloc(sizeof(int) * input.thread_count));
        instance.count_historic_weights = (int*)(malloc(sizeof(int) * CMOCHC_PARETO_FRONT__PATCHES));
        for (int t = 0; t < CMOCHC_PARETO_FRONT__PATCHES; t++) {
            instance.count_historic_weights[t] = 0;
        }
    #endif

    /* Sync */
    if (pthread_barrier_init(&(instance.sync_barrier), NULL, input.thread_count + 1))
    {
        fprintf(stderr, "[ERROR] could not create a sync barrier.\n");
        exit(EXIT_FAILURE);
    }

    /* Estado de la población elite global */
    instance.iter_elite_pop = (struct solution*)(malloc(sizeof(struct solution) * (input.thread_count * CMOCHC_LOCAL__BEST_SOLS_KEPT)));
    instance.iter_elite_pop_tag = (int*)(malloc(sizeof(int) * (input.thread_count * CMOCHC_LOCAL__BEST_SOLS_KEPT)));

    for (int i = 0; i < (input.thread_count * CMOCHC_LOCAL__BEST_SOLS_KEPT); i++) {
        create_empty_solution(&instance.iter_elite_pop[i], &current_scenario, &etc, &energy);
    }

    /* Inicializo los hilos */
    instance.threads = (pthread_t*)malloc(sizeof(pthread_t) * input.thread_count);
    *threads_data = (struct cmochc_thread*)malloc(sizeof(struct cmochc_thread) * input.thread_count);

    for (int i = 0; i < input.thread_count; i++)
    {
        pthread_t *t;
        t = &(instance.threads[i]);

        struct cmochc_thread *t_data;
        t_data = &((*threads_data)[i]);
        t_data->thread_id = i;
        t_data->data = &instance;

        if (pthread_create(t, NULL, slave_thread, (void*) t_data))
        {
            fprintf(stderr, "[ERROR] could not create slave thread %d\n", i);
            exit(EXIT_FAILURE);
        }
    }

    /* Inicializo el archivador */
    archivers_aga_init(&instance.archiver, CMOCHC_ARCHIVE__MAX_SIZE, instance.iter_elite_pop, 
        instance.iter_elite_pop_tag, (input.thread_count * CMOCHC_LOCAL__BEST_SOLS_KEPT), 
        CMOCHC_PARETO_FRONT__PATCHES);
}

/* Obtiene los mejores elementos de cada población */
int gather(struct cmochc &instance) {
    #ifdef DEBUG_3
        fprintf(stderr, "[DEBUG] Gathering...\n");
        fprintf(stderr, "[DEBUG] Current iteration elite solutions:\n");

        int cantidad = 0;
        for (int i = 0; i < instance.archiver.new_solutions_size; i++) {
            if (instance.archiver.new_solutions[i].initialized == 1) cantidad++;

            fprintf(stderr, "> %d state=%d makespan=%f energy=%f\n",
                i, instance.archiver.new_solutions[i].initialized,
                instance.archiver.new_solutions[i].makespan,
                instance.archiver.new_solutions[i].energy_consumption);
        }

        ASSERT(cantidad > 0);
    #endif

    int new_solutions;
    new_solutions = archivers_aga(&instance.archiver);

    #ifdef DEBUG_3
        fprintf(stderr, "[DEBUG] Total solutions gathered      = %d\n", new_solutions);
        fprintf(stderr, "[DEBUG] Current solutions in archiver = %d\n", instance.archiver.population_count);
    #endif
    
    #ifdef DEBUG_1
        int current_tag;
        for (int s = 0; s < instance.archiver.population_size; s++) {
            if (instance.archiver.population[s].initialized == SOLUTION__IN_USE) {
                current_tag = instance.archiver.population_tag[s];
                instance.count_historic_weights[current_tag]++;
            }
        }
    #endif
    
    return new_solutions;
}

/* Logica de los esclavos */
void* slave_thread(void *data) {
    struct cmochc_thread *t_data = (struct cmochc_thread*)data;
    struct cmochc *instance = t_data->data;

    int thread_id = t_data->thread_id;

    struct params *input = instance->input;
    struct scenario *current_scenario = instance->current_scenario;
    struct etc_matrix *etc = instance->etc;
    struct energy_matrix *energy = instance->energy;
       
    FLOAT *weights_array = instance->weights;
    int *thread_weight_assignment = instance->thread_weight_assignment;
    //int *weight_thread_assignment = instance->weight_thread_assignment;

    RAND_STATE *rand_state = instance->rand_state;

    int *count_generations = instance->count_generations;
    int *count_at_least_one_children_inserted = instance->count_at_least_one_children_inserted;
    int *count_improved_best_sol = instance->count_improved_best_sol;
    int *count_crossover = instance->count_crossover;
    int *count_improved_crossover = instance->count_improved_crossover;
    int *count_cataclysm = instance->count_cataclysm;
    int *count_improved_mutation = instance->count_improved_mutation;
    int *count_migrations = instance->count_migrations;
    int *count_solutions_migrated = instance->count_solutions_migrated;

    /* *********************************************************************************************
     * Inicializo el thread.
     * *********************************************************************************************/

    count_generations[thread_id] = 0;
    count_at_least_one_children_inserted[thread_id] = 0;
    count_improved_best_sol[thread_id] = 0;
    count_crossover[thread_id] = 0;
    count_improved_crossover[thread_id] = 0;
    count_cataclysm[thread_id] = 0;
    count_improved_mutation[thread_id] = 0;
    count_migrations[thread_id] = 0;
    count_solutions_migrated[thread_id] = 0;

    /* Inicialización del estado del generador aleatorio */
    RAND_INIT(thread_id, rand_state[thread_id]);
    FLOAT random;

    /* *********************************************************************************************
     * Inicializo los pesos.
     * *********************************************************************************************/

    int currently_assigned_weight = thread_weight_assignment[thread_id];
    FLOAT weight_makespan = weights_array[thread_weight_assignment[thread_id]];
    FLOAT energy_makespan = 1 - weight_makespan;

    /* *********************************************************************************************
     * Inicializo la población.
     * *********************************************************************************************/

    /* Inicializo la población de padres y limpio la de hijos */
    int max_pop_sols = 2 * input->population_size;

    /* Poblacion de cada esclavo */
    struct solution *population = (struct solution*)(malloc(sizeof(struct solution) * max_pop_sols));
    int *sorted_population = (int*)(malloc(sizeof(int) * max_pop_sols));
    FLOAT *fitness_population = (FLOAT*)(malloc(sizeof(FLOAT) * max_pop_sols));

    //int makespan_utopia_index, energy_utopia_index;
    FLOAT makespan_utopia_value, energy_utopia_value;
    FLOAT makespan_nadir_value, energy_nadir_value;

    for (int i = 0; i < max_pop_sols; i++) {
        // Random init.
        create_empty_solution(&(population[i]),current_scenario,etc,energy);

        random = RAND_GENERATE(rand_state[thread_id]);
        int starting_pos;
        starting_pos = (int)(floor(etc->tasks_count * random));

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
                mutate(rand_state[thread_id], &population[i], &population[i]);
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

    /* Merge sort tmp array */
    int *merge_sort_tmp = (int*)malloc(sizeof(int) * max_pop_sols);

    merge_sort(population, weight_makespan, energy_makespan, makespan_utopia_value, energy_utopia_value,
        makespan_nadir_value, energy_nadir_value, sorted_population, fitness_population, max_pop_sols,
        merge_sort_tmp);

    /* *********************************************************************************************
     * Main iteration
     * ********************************************************************************************* */
    int next_avail_children;
    int max_children = input->population_size / 2;
    int max_distance = etc->tasks_count;

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

    while (instance->stopping_condition == 0) {
        for (int iteracion = 0; iteracion < CMOCHC_LOCAL__ITERATION_COUNT; iteracion++) {
            #ifdef DEBUG_1
                count_generations[thread_id]++;
            #endif

            /* *********************************************************************************************
             * Mating
             * ********************************************************************************************* */
            next_avail_children = input->population_size;

            FLOAT d;
            int p1_idx, p2_idx;
            int p1_rand, p2_rand;
            int c1_idx, c2_idx;
            for (int child = 0; child < max_children; child++) {
                if (next_avail_children + 1 < max_pop_sols) {
                    // Padre aleatorio 1
                    random = RAND_GENERATE(rand_state[thread_id]);
                    p1_rand = (int)(floor(input->population_size * random));
                    p1_idx = sorted_population[p1_rand];

                    // Padre aleatorio 2
                    random = RAND_GENERATE(rand_state[thread_id]);
                    p2_rand = (int)(floor((input->population_size - 1) * random));
                    if (p2_rand >= p1_rand) p2_rand++;
                    p2_idx = sorted_population[p2_rand];

                    // Chequeo la distancia entre padres
                    d = distance(&population[p1_idx],&population[p2_idx]);

                    if (d > threshold) {
                        // Aplico HUX y creo dos hijos
                        count_crossover[thread_id]++;

                        c1_idx = sorted_population[next_avail_children];
                        c2_idx = sorted_population[next_avail_children+1];

                        hux(rand_state[thread_id],
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

                                count_improved_crossover[thread_id]++;
                            }
                        #endif

                        next_avail_children += 2;
                    }
                }
            }

            if (next_avail_children > input->population_size) {
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
                    energy_nadir_value, sorted_population[input->population_size-1]);

                merge_sort(population, weight_makespan, energy_makespan, makespan_utopia_value, energy_utopia_value,
                    makespan_nadir_value, energy_nadir_value, sorted_population,
                    fitness_population, max_pop_sols, merge_sort_tmp);

                if (worst_parent > fitness(population, fitness_population, weight_makespan, energy_makespan,
                        makespan_utopia_value, energy_utopia_value, makespan_nadir_value,
                        energy_nadir_value, sorted_population[input->population_size-1])) {

                    #ifdef DEBUG_1
                        count_at_least_one_children_inserted[thread_id]++;
                    #endif
                } else {
                    threshold -= threshold_step;
                }

                if (best_parent > fitness(population, fitness_population, weight_makespan, energy_makespan,
                        makespan_utopia_value, energy_utopia_value, makespan_nadir_value,
                        energy_nadir_value, sorted_population[0])) {

                    #ifdef DEBUG_1
                        count_improved_best_sol[thread_id]++;
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
                            count_cataclysm[thread_id]++;
                            pre_mut_fitness = fitness(population, fitness_population, weight_makespan, energy_makespan, 
                                makespan_utopia_value, energy_utopia_value, makespan_nadir_value, energy_nadir_value, 
                                sorted_population[i]);
                        #endif

                        aux_index = RAND_GENERATE(rand_state[thread_id]) * CMOCHC_LOCAL__BEST_SOLS_KEPT;

                        mutate(rand_state[thread_id],
                            &population[sorted_population[aux_index]],
                            &population[sorted_population[i]]);

                        fitness_population[sorted_population[i]] = NAN;
                        fitness(population, fitness_population, weight_makespan, energy_makespan, makespan_utopia_value, 
                            energy_utopia_value, makespan_nadir_value, energy_nadir_value, sorted_population[i]);

                        #ifdef DEBUG_1
                            if (fitness_population[sorted_population[i]] < pre_mut_fitness) {
                                count_improved_mutation[thread_id]++;
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
        
            clone_solution(&instance->iter_elite_pop[iter_elite_pop_index], &population[local_best_index]);
            instance->iter_elite_pop_tag[iter_elite_pop_index] = thread_weight_assignment[thread_id];
        }

        /* Le aviso al maestro que puede empezar con la operación de gather. */
        rc = pthread_barrier_wait(&instance->sync_barrier);
        if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
        {
            printf("Could not wait on barrier\n");
            exit(EXIT_FAILURE);
        }

        // ..................

        /* Espero a que el maestro ejecute la operación de gather. */
        rc = pthread_barrier_wait(&instance->sync_barrier);
        if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
        {
            printf("Could not wait on barrier\n");
            exit(EXIT_FAILURE);
        }

        if (instance->stopping_condition == 0) {
            /* Actualizo la nueva asignación de pesos y vecindad */
            #ifdef DEBUG_3
                fprintf(stderr, "[DEBUG] Thread %d, Current weights=%d (%.4f,%.4f), new weights=%d (%.4f,%.4f)\n",
                    thread_id, currently_assigned_weight, weight_makespan, energy_makespan, 
                    thread_weight_assignment[thread_id], weights_array[thread_weight_assignment[thread_id]], 
                    1 - weights_array[thread_weight_assignment[thread_id]]);
            #endif
            
            //int changed_assignment;
            
            if (currently_assigned_weight != thread_weight_assignment[thread_id]) {
                //changed_assignment = 1;
                
                weight_makespan = weights_array[thread_weight_assignment[thread_id]];
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
                            
                    while ((ep_current_index < instance->archiver.population_size) &&
                        (ep_solution_index < instance->archiver.population_count)) {
                            
                        if (instance->archiver.population[ep_current_index].initialized == SOLUTION__IN_USE) {
                            if (ep_solution_index < CMOCHC_COLLABORATION__MOEAD_NEIGH_SIZE) {
                                /* Aún no esta lleno el array con el vecindario */
                                migration_global_pop_index[ep_solution_index] = ep_current_index;
                                migration_current_weight_distance[ep_solution_index] = abs(thread_weight_assignment[thread_id] -
                                    instance->archiver.population_tag[ep_current_index]);
                                    
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
                                current_solution_distance = abs(thread_weight_assignment[thread_id] -
                                    instance->archiver.population_tag[ep_current_index]);
                                    
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
                    selection_prob = 1 / instance->archiver.population_count;
                
                    while ((ep_current_index < instance->archiver.population_size) &&
                        (ep_solution_index < instance->archiver.population_count) &&
                        (current_ep_solution_index < CMOCHC_COLLABORATION__MOEAD_NEIGH_SIZE)) {
                            
                        if (instance->archiver.population[ep_current_index].initialized == SOLUTION__IN_USE) {
                            if (RAND_GENERATE(rand_state[thread_id]) <= selection_prob) {
                                /* Aún no esta lleno el array con el vecindario */
                                migration_global_pop_index[current_ep_solution_index] = ep_current_index;
                                migration_current_weight_distance[current_ep_solution_index] = 
                                    abs(thread_weight_assignment[thread_id] -
                                        instance->archiver.population_tag[ep_current_index]);
                                
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
                            migration_global_pop_index[i] = -1;
                            
                            fprintf(stderr, "[DEBUG] Thread %d, Neighbour %d (weight idx %d, distance %d).\n", 
                                thread_id, 
                                migration_global_pop_index[i], 
                                instance->archiver.population_tag[migration_global_pop_index[i]],
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
                            count_migrations[thread_id]++;
                        
                            #ifdef CMOCHC_COLLABORATION__MIGRATE_BY_COPY
                                clone_solution(&population[next_solution_index],
                                    &instance->archiver.population[migration_global_pop_index[migrated_solution_index]]);
                                    
                                next_solution_index++;
                                migrated_solution_index++;
                                count_solutions_migrated[thread_id]++;
                                migrated = 1;
                            #endif
                            #ifdef CMOCHC_COLLABORATION__MIGRATE_BY_MATE
                                if (next_solution_index + 1 < max_pop_sols) {
                                    random = RAND_GENERATE(rand_state[thread_id]);
                                    
                                    int migration_parent_index;
                                    migration_parent_index = (int)(random * next_solution_index);
                                    
                                    FLOAT d = distance(
                                        &instance->archiver.population[migration_global_pop_index[migrated_solution_index]],
                                        &population[migration_parent_index]);

                                    if (d > threshold_max) {
                                        hux(rand_state[thread_id],
                                            &instance->archiver.population[migration_global_pop_index[migrated_solution_index]], 
                                            &population[migration_parent_index],
                                            &population[sorted_population[next_solution_index]],
                                            &population[sorted_population[next_solution_index+1]]);
                                            
                                        next_solution_index += 2;
                                        migrated_solution_index++;
                                        count_solutions_migrated[thread_id]++;
                                        migrated = 1;
                                    }
                                }
                                
                                if (migrated == 0) {
                                    mutate(
                                        rand_state[thread_id],
                                        &instance->archiver.population[migration_global_pop_index[migrated_solution_index]],
                                        &population[next_solution_index]);
                                        
                                    next_solution_index++;
                                    migrated_solution_index++;
                                    count_solutions_migrated[thread_id]++; 
                                    migrated = 1;
                                }
                            #endif
                            #ifdef CMOCHC_COLLABORATION__MIGRATE_BY_MUTATE
                                mutate(
                                    rand_state[thread_id],
                                    &instance->archiver.population[migration_global_pop_index[migrated_solution_index]],
                                    &population[next_solution_index]);
                                    
                                next_solution_index++;
                                migrated_solution_index++;
                                count_solutions_migrated[thread_id]++;
                                
                                migrated = 1;
                            #endif
                        }
                    }
                #endif
                
                if (migrated == 0) {
                    random = RAND_GENERATE(rand_state[thread_id]);

                    mutate(rand_state[thread_id],
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
                for (int i = 0; i < input->population_size; i++) {
                    fprintf(stderr, "%d(%f)<%d>  ", sorted_population[i], 
                        fitness_population[sorted_population[i]], 
                        population[sorted_population[i]].initialized);
                }
                fprintf(stderr, "\n");
                fprintf(stderr, "childs > ");
                for (int i = input->population_size; i < max_pop_sols; i++) {
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

    RAND_FINALIZE(rand_state[thread_id]);
    
    for (int p = 0; p < max_pop_sols; p++) {
        free_solution(&population[p]);
    }
    
    free(merge_sort_tmp);
    free(population);
    free(sorted_population);
    free(fitness_population);

    return 0;
}

void display_results(struct cmochc &instance) {
    /* Show solutions */
    #if defined(OUTPUT_SOLUTION)
        archivers_aga_dump(&instance.archiver);
    #endif

    #ifdef DEBUG_1
        archivers_aga_show(&instance.archiver);

        int count_generations = 0;
        int count_at_least_one_children_inserted = 0;
        int count_improved_best_sol = 0;
        int count_crossover = 0;
        int count_improved_crossover = 0;
        int count_cataclysm = 0;
        int count_improved_mutation = 0;
        int count_migrations = 0;
        int count_solutions_migrated = 0;

        for (int t = 0; t < instance.input->thread_count; t++) {
            count_generations += instance.count_generations[t];
            count_at_least_one_children_inserted += instance.count_at_least_one_children_inserted[t];
            count_improved_best_sol += instance.count_improved_best_sol[t];
            count_crossover += instance.count_crossover[t];
            count_improved_crossover += instance.count_improved_crossover[t];
            count_cataclysm += instance.count_cataclysm[t];
            count_improved_mutation += instance.count_improved_mutation[t];
            count_migrations += instance.count_migrations[t];
            count_solutions_migrated += instance.count_solutions_migrated[t];
        }

        int *count_pf_found;
        count_pf_found = (int*)(malloc(sizeof(int) * CMOCHC_PARETO_FRONT__PATCHES));

        for (int s = 0; s < CMOCHC_PARETO_FRONT__PATCHES; s++) {
            count_pf_found[s] = 0;
        }

        for (int s = 0; s < instance.archiver.population_size; s++) {
            if (instance.archiver.population[s].initialized == SOLUTION__IN_USE) {
                count_pf_found[instance.archiver.population_tag[s]]++;
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
            fprintf(stderr, "          [%d] = %d\n", t, instance.archiver.tag_count[t]);
        }

        fprintf(stderr, "       historic archive tag count:\n");
        for (int t = 0; t < CMOCHC_PARETO_FRONT__PATCHES; t++) {
            fprintf(stderr, "          [%d] = %d (%d)\n", t, count_pf_found[t], instance.count_historic_weights[t]);
        }

        fprintf(stderr, "[INFO] ========================================================\n");
        
        free(count_pf_found);
    #endif
}

/* Libera los recursos pedidos y finaliza la ejecución */
void finalize(struct cmochc &instance, struct cmochc_thread *threads) {
    archivers_aga_free(&instance.archiver);
    pthread_barrier_destroy(&(instance.sync_barrier));

    free(instance.count_generations);
    free(instance.count_at_least_one_children_inserted);
    free(instance.count_improved_best_sol);
    free(instance.count_crossover);
    free(instance.count_improved_crossover);
    free(instance.count_cataclysm);
    free(instance.count_improved_mutation);
    free(instance.count_migrations);
    free(instance.count_solutions_migrated);
    free(instance.count_historic_weights);
    
    free(instance.weights);
    free(instance.thread_weight_assignment);
    free(instance.weight_thread_assignment);
        
    for (int i = 0; i < (instance.input->thread_count * CMOCHC_LOCAL__BEST_SOLS_KEPT); i++) {
        free_solution(&instance.iter_elite_pop[i]);
    }
        
    free(instance.iter_elite_pop);
    free(instance.iter_elite_pop_tag);
    
    free(instance.rand_state);
    free(instance.threads);

    free(threads);
}
