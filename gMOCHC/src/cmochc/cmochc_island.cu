#include <pthread.h>
#include <math.h>

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
    
    FLOAT random;
    
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

        for (int i = 0; i < input.thread_count; i++) {
            random = RAND_GENERATE(rstate);
            instance.thread_weight_assignment[i] = (int)(random * CMOCHC_PARETO_FRONT__PATCHES);
        }

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
    fprintf(stderr, "       CMOCHC_PARETO_FRONT__SYNC_WEIGHT_ASSIGN     : ");
    #ifdef CMOCHC_PARETO_FRONT__SYNC_WEIGHT_ASSIGN
        fprintf(stderr, "YES\n");
    #else
        fprintf(stderr, "NO\n");
    #endif
    fprintf(stderr, "       CMOCHC_ARCHIVE__MAX_SIZE                    : %d\n", CMOCHC_ARCHIVE__MAX_SIZE);
    
    
    /*fprintf(stderr, "       CMOCHC_COLLABORATION__MOEAD_NEIGH_SIZE      : %d\n", CMOCHC_COLLABORATION__MOEAD_NEIGH_SIZE);
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
    fprintf(stderr, "       CMOCHC_COLLABORATION__MUTATE                : ");
    #ifdef CMOCHC_COLLABORATION__MUTATE_BEST
        fprintf(stderr, "BEST\n");
    #endif
    #ifdef CMOCHC_COLLABORATION__MUTATE_ALL_ELITE
        fprintf(stderr, "ALL_ELITE\n");
    #endif
    #ifdef CMOCHC_COLLABORATION__MUTATE_NONE
        fprintf(stderr, "NONE\n");
    #endif*/
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
    
    int *thread_weight_assignment;
    int *weight_thread_assignment;

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
        instance.iter_elite_pop_tag, (input.thread_count * CMOCHC_LOCAL__BEST_SOLS_KEPT));
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
    int *weight_thread_assignment = instance->weight_thread_assignment;

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

    int makespan_utopia_index, energy_utopia_index;
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
            makespan_utopia_index = i;

            energy_utopia_value = population[i].energy_consumption;
            energy_nadir_value = energy_utopia_value;
            energy_utopia_index = i;
        } else {
            #ifdef CMOCHC_LOCAL__MUTATE_INITIAL_POP
                mutate(rand_state[thread_id], &population[i], &population[i]);
            #endif

            if (population[i].makespan < makespan_utopia_value) {
                makespan_utopia_index = i;
                makespan_utopia_value = population[i].makespan;

                if (population[i].energy_consumption > energy_nadir_value) {
                    energy_nadir_value = population[i].energy_consumption;
                }
            }
            if (population[i].energy_consumption < energy_utopia_value) {
                energy_utopia_index = i;
                energy_utopia_value = population[i].energy_consumption;

                if (population[i].makespan > makespan_nadir_value) {
                    makespan_nadir_value = population[i].makespan;
                }
            }
        }
    }

    #ifdef DEBUG_3
        fprintf(stderr, "[DEBUG] Normalization references\n");
        fprintf(stderr, "> (makespan) best index=%d, zenith=%.2f, nadir=%.2f\n",
            makespan_utopia_index, makespan_utopia_value, makespan_nadir_value);
        fprintf(stderr, "> (energy) best index=%d, zenith=%.2f, nadir=%.2f\n",
            energy_utopia_index, energy_utopia_value, energy_nadir_value);
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
        makespan_nadir_value, energy_nadir_value, sorted_population,
        fitness_population, max_pop_sols);

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
                    fitness_population, max_pop_sols);

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
                    fitness_population, max_pop_sols);
            }
        }

        /* *********************************************************************************************
         * Fin de iteracion local
         * ********************************************************************************************* */

        /* Copio mis mejores soluciones a la población de intercambio principal */
        int iter_elite_pop_index;
        int local_best_index;

        #ifdef DEBUG_3
            fprintf(stderr, "[DEBUG] Pre-copy population\n");
            fprintf(stderr, "best>\n");
            for (int i = 0; i < CMOCHC_LOCAL__BEST_SOLS_KEPT; i++) {
                fprintf(stderr, "%d[%.4f,%.4f]<%d>\n", sorted_population[i], 
                    population[sorted_population[i]].makespan, 
                    population[sorted_population[i]].energy_consumption, 
                    population[sorted_population[i]].initialized);
            }
        #endif
        
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
        
        #ifdef DEBUG_3
            fprintf(stderr, "[DEBUG] Post-copy population\n");
            fprintf(stderr, "local_best>\n");
            for (int i = 0; i < input->thread_count * CMOCHC_LOCAL__BEST_SOLS_KEPT; i++) {
                fprintf(stderr, "%d[%.4f,%.4f]<%d>\n", i, 
                    instance->iter_elite_pop[i].makespan, 
                    instance->iter_elite_pop[i].energy_consumption, 
                    instance->iter_elite_pop[i].initialized);
            }
        #endif
        
        #ifdef DEBUG_3
            fprintf(stderr, "[DEBUG] Post-migration population\n");
            fprintf(stderr, "parents> ");
            for (int i = 0; i < input->population_size; i++) {
                fprintf(stderr, "%d(%f)<%d>  ", sorted_population[i], fitness_population[sorted_population[i]], population[sorted_population[i]].initialized);
            }
            fprintf(stderr, "\n");
            fprintf(stderr, "childs > ");
            for (int i = input->population_size; i < max_pop_sols; i++) {
                fprintf(stderr, "%d(%f)<%d>  ", sorted_population[i], fitness_population[sorted_population[i]], population[sorted_population[i]].initialized);
            }
            fprintf(stderr, "\n");
        #endif

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
            
            int changed_assignment;
            
            if (currently_assigned_weight != thread_weight_assignment[thread_id]) {
                changed_assignment = 1;
                
                weight_makespan = weights_array[thread_weight_assignment[thread_id]];
                energy_makespan = 1 - weight_makespan;
            } else {
                changed_assignment = 0;
            }

            /* *********************************************************************************************
             * Busco las mejores soluciones elite a importar
             * ********************************************************************************************* */
            int best_index[3];
            best_index[0] = -1;
            best_index[1] = -1;
            best_index[2] = -1;
            
            int current_index, sol_index;
            current_index = 0;
            sol_index = 0;
                    
            while ((current_index < instance->archiver.population_size)&&
                (sol_index < instance->archiver.population_count)) {
                    
                if (instance->archiver.population[current_index].initialized == SOLUTION__IN_USE) {
                    if (instance->archiver.population_tag[current_index] < thread_weight_assignment[thread_id]) {
                        if (best_index[0] == -1) {
                            best_index[0] = current_index;
                        } else if (instance->archiver.population_tag[current_index] > instance->archiver.population_tag[best_index[0]]) {
                            best_index[0] = current_index;
                        }
                    } else if (instance->archiver.population_tag[current_index] > thread_weight_assignment[thread_id]) {
                        if (best_index[1] == -1) {
                            best_index[1] = current_index;
                        } else if (instance->archiver.population_tag[current_index] < instance->archiver.population_tag[best_index[1]]) {
                            best_index[1] = current_index;
                        }                    
                    } else if (instance->archiver.population_tag[current_index] == thread_weight_assignment[thread_id]) {
                        best_index[2] = current_index;
                    }
                    
                    sol_index++;
                }
                
                current_index++;
            }
            
            #ifdef DEBUG_3        
                if (best_index[0] != -1) {
                    fprintf(stderr, "[DEBUG] Thread %d, Best previous solution found in %d (weight %d).\n", thread_id, 
                        best_index[0], instance->archiver.population_tag[best_index[0]]);
                } else {
                    fprintf(stderr, "[DEBUG] Thread %d, No best previous solution found.\n", thread_id);
                }
                if (best_index[2] != -1) {
                    fprintf(stderr, "[DEBUG] Thread %d, Best solution found in %d (weight %d).\n", thread_id, 
                        best_index[2], instance->archiver.population_tag[best_index[2]]);
                } else {
                    fprintf(stderr, "[DEBUG] Thread %d, No best solution found.\n", thread_id);
                }
                if (best_index[1] != -1) {
                    fprintf(stderr, "[DEBUG] Thread %d, Best next solution found in %d (weight %d).\n", thread_id, 
                        best_index[1], instance->archiver.population_tag[best_index[1]]);
                } else {
                    fprintf(stderr, "[DEBUG] Thread %d, No best next solution found.\n", thread_id);
                }            
            #endif

            /* *********************************************************************************************
             * Migro soluciones desde la población elite
             * ********************************************************************************************* */
             
            int migrated, next_solution, source_index;
            next_solution = 0;
            migrated = 0;
            
            for (int i = CMOCHC_LOCAL__BEST_SOLS_KEPT; i < max_pop_sols; i++) {
                source_index = i - CMOCHC_LOCAL__BEST_SOLS_KEPT;
                
                if (source_index < 3) {
                    if (best_index[source_index] != -1) {
                        #ifndef CMOCHC_COLLABORATION__MIGRATION_NONE
                            count_migrations[thread_id]++;
                        
                            #ifdef CMOCHC_COLLABORATION__MIGRATE_BY_COPY
                                clone_solution(&population[sorted_population[next_solution]],&instance->archiver.population[best_index[source_index]]);
                                next_solution++;
                                count_solutions_migrated[thread_id]++;
                            #endif
                            #ifdef CMOCHC_COLLABORATION__MIGRATE_BY_MATE
                                if (next_solution + 1 < max_pop_sols) {
                                    FLOAT d = distance(&instance->archiver.population[best_index[source_index]],&population[sorted_population[0]]);

                                    if (d > threshold_max) {
                                        hux(rand_state[thread_id],
                                            &instance->archiver.population[best_index[source_index]], &population[sorted_population[0]],
                                            &population[sorted_population[next_solution]],&population[sorted_population[next_solution+1]]);
                                    }

                                    next_solution += 2;
                                    count_solutions_migrated[thread_id]++;
                                }
                            #endif
                            #ifdef CMOCHC_COLLABORATION__MIGRATE_BY_MUTATE
                                mutate(rand_state[thread_id],&instance->archiver.population[best_index[source_index]],&population[sorted_population[next_solution]]);
                                next_solution++;
                                count_solutions_migrated[thread_id]++;
                            #endif

                            migrated = 1;
                        #endif
                    }
                }
                
                if (migrated == 0) {
                    random = RAND_GENERATE(rand_state[thread_id]);

                    mutate(rand_state[thread_id],
                        &population[sorted_population[(int)(random * CMOCHC_LOCAL__BEST_SOLS_KEPT)]],
                        &population[sorted_population[next_solution]]);

                    next_solution++;
                }
            }
            
            /* Actualizo los puntos de normalización con la población local */
            for (int i = 0; i < max_pop_sols; i++) {
                if (population[i].makespan < makespan_utopia_value) {
                    makespan_utopia_index = i;
                    makespan_utopia_value = population[i].makespan;

                    if (population[i].energy_consumption > energy_nadir_value) {
                        energy_nadir_value = population[i].energy_consumption;
                    }
                }
                if (population[i].energy_consumption < energy_utopia_value) {
                    energy_utopia_index = i;
                    energy_utopia_value = population[i].energy_consumption;

                    if (population[i].makespan > makespan_nadir_value) {
                        makespan_nadir_value = population[i].makespan;
                    }
                }
            }

            /* *********************************************************************************************
             * Muto las soluciones no migradas?
             *********************************************************************************************** */
            #ifdef CMOCHC_COLLABORATION__MUTATE_BEST
                for (int i = 0; i < 1; i++) {
                    mutate(rand_state[thread_id], &population[sorted_population[i]], &population[sorted_population[i]]);
                }
            #endif
            #ifdef CMOCHC_COLLABORATION__MUTATE_ALL_ELITE
                for (int i = 0; i < CMOCHC_LOCAL__BEST_SOLS_KEPT; i++) {
                    mutate(rand_state[thread_id], &population[sorted_population[i]], &population[sorted_population[i]]);
                }
            #endif
            #ifdef CMOCHC_COLLABORATION__MUTATE_BEST
            #endif

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
                fitness_population, max_pop_sols);

            #ifdef DEBUG_3
                fprintf(stderr, "[DEBUG] Post-migration population\n");
                fprintf(stderr, "parents> ");
                for (int i = 0; i < input->population_size; i++) {
                    fprintf(stderr, "%d(%f)<%d>  ", sorted_population[i], fitness_population[sorted_population[i]], population[sorted_population[i]].initialized);
                }
                fprintf(stderr, "\n");
                fprintf(stderr, "childs > ");
                for (int i = input->population_size; i < max_pop_sols; i++) {
                    fprintf(stderr, "%d(%f)<%d>  ", sorted_population[i], fitness_population[sorted_population[i]], population[sorted_population[i]].initialized);
                }
                fprintf(stderr, "\n");
            #endif
        }
    }

    // ================================================================
    // Finalizo el thread.
    // ================================================================   

    RAND_FINALIZE(rand_state[thread_id]);
    
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

        int *count_pf_found;
        count_pf_found = (int*)(malloc(sizeof(int) * CMOCHC_PARETO_FRONT__PATCHES));

        for (int t = 0; t < instance.input->thread_count; t++) {
            count_pf_found[t] = 0;

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

        fprintf(stderr, "       pf solution count by deme:\n");
        for (int t = 0; t < instance.input->thread_count; t++) {
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
    free(instance.weights);
        
    free(instance.iter_elite_pop);
    free(instance.iter_elite_pop_tag);
    
    free(instance.rand_state);
    free(instance.threads);

    free(threads);
}
