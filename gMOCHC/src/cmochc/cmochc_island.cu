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

struct cmochc {
    struct params *input;
    struct scenario *current_scenario;
    struct etc_matrix *etc;
    struct energy_matrix *energy;

    /* Coleccion de esclavos */
    pthread_t *threads;

    /* Poblacion de cada esclavo */
    struct solution **population;
    int **sorted_population;

    /* Poblacion elite global mantenida por el master */
    struct solution *global_elite_pop;
    struct aga_state archiver;

    float **weights;
    int stopping_condition;

    /* Random generator de cada esclavo */
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
void gather(struct cmochc &instance);

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
    for (int iteracion = 0; iteracion < input.max_iterations; iteracion++) {
        /* ************************************************** */
        /* Espero a que los esclavos terminen de evolucionar. */
        /* ************************************************** */
        rc = pthread_barrier_wait(&instance.sync_barrier);
        if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
        {
            printf("Could not wait on barrier\n");
            exit(EXIT_FAILURE);
        }

        /* Los esclavos copian sus mejores soluciones y comienzan a migrar */

        if (iteracion + 1 >= input.max_iterations) {
            /* Si esta es la úlitma iteracion, les aviso a los esclavos */
            instance.stopping_condition = 1;
        }

        /* ************************************************ */
        /* Espero que los esclavos terminen el intercambio. */
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

        gather(instance);

        TIMMING_END(">> cmochc_gather", ts_gather);

        #ifdef CMOCHC_SYNC
            /* Notifico a los esclavos que termió la operación de gather. */
            rc = pthread_barrier_wait(&instance.sync_barrier);
            if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
            {
                printf("Could not wait on barrier\n");
                exit(EXIT_FAILURE);
            }
        #endif
    }

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
    finalize(instance, threads);
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
        count_pf_found = (int*)(malloc(sizeof(int) * instance.input->thread_count));

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
                count_pf_found[instance.archiver.population_origin[s] / CMOCHC_LOCAL__BEST_SOLS_KEPT]++;
            }
        }

        fprintf(stderr, "[INFO] == Statistics ==========================================\n");
        fprintf(stderr, "       count_generations                    : %d\n", count_generations);
        fprintf(stderr, "       count_at_least_one_children_inserted : %d (%.2f %%)\n", count_at_least_one_children_inserted,
            ((float)count_at_least_one_children_inserted/(float)count_generations)*100);
        fprintf(stderr, "       count_improved_best_sol              : %d (%.2f %%)\n", count_improved_best_sol,
            ((float)count_improved_best_sol/(float)count_generations)*100);
        fprintf(stderr, "       count_crossover                      : %d\n", count_crossover);
        fprintf(stderr, "       count_improved_crossover             : %d (%.2f %%)\n", count_improved_crossover,
            ((float)count_improved_crossover/(float)count_crossover)*100);
        fprintf(stderr, "       count_cataclysm                      : %d\n", count_cataclysm);
        fprintf(stderr, "       count_improved_mutation              : %d (%.2f %%)\n", count_improved_mutation,
            ((float)count_improved_mutation/(float)count_cataclysm)*100);
        fprintf(stderr, "       count_migrations                     : %d\n", count_migrations);
        fprintf(stderr, "       count_solutions_migrated             : %d (%.2f %%)\n", count_solutions_migrated,
            ((float)count_solutions_migrated/(float)count_migrations)*100);

        fprintf(stderr, "       pf solution count by deme:\n");
        for (int t = 0; t < instance.input->thread_count; t++) {
            fprintf(stderr, "          [%d] = %d (%d)\n", t, count_pf_found[t], instance.count_historic_weights[t]);
        }

        fprintf(stderr, "[INFO] ========================================================\n");
    #endif
}

/* Inicializa los hilos y las estructuras de datos */
void init(struct cmochc &instance, struct cmochc_thread **threads_data,
    struct params &input, struct scenario &current_scenario,
    struct etc_matrix &etc, struct energy_matrix &energy) {

    fprintf(stderr, "[INFO] == Global configuration constants ======================\n");
    fprintf(stderr, "       CMOCHC_LOCAL__ITERATION_COUNT               : %d\n", CMOCHC_LOCAL__ITERATION_COUNT);
    fprintf(stderr, "       CMOCHC_LOCAL__BEST_SOLS_KEPT                : %d\n", CMOCHC_LOCAL__BEST_SOLS_KEPT);
    fprintf(stderr, "       CMOCHC_LOCAL__MATING_MAX_THRESHOLD_DIVISOR  : %d\n", CMOCHC_LOCAL__MATING_MAX_THRESHOLD_DIVISOR);
    fprintf(stderr, "       CMOCHC_LOCAL__MATING_THRESHOLD_STEP_DIVISOR : %d\n", CMOCHC_LOCAL__MATING_THRESHOLD_STEP_DIVISOR);


    fprintf(stderr, "       CMOCHC_LOCAL__MUTATE_INITIAL_POP :");
    #ifdef CMOCHC_LOCAL__MUTATE_INITIAL_POP
        fprintf(stderr, " YES\n");
    #else
        fprintf(stderr, " NO\n");
    #endif

    fprintf(stderr, "       CMOCHC_LOCAL__FITNESS_NORM : ");
    #ifdef CMOCHC_LOCAL__Z_FITNESS_NORM
        fprintf(stderr, " Z\n");
    #endif
    #ifdef CMOCHC_LOCAL__ZN_FITNESS_NORM
        fprintf(stderr, " ZN\n");
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
    fprintf(stderr, "       CMOCHC_COLLABORATION__MUTATE                : ");
    #ifdef CMOCHC_COLLABORATION__MUTATE_BEST
        fprintf(stderr, "BEST\n");
    #endif
    #ifdef CMOCHC_COLLABORATION__MUTATE_ALL_ELITE
        fprintf(stderr, "ALL_ELITE\n");
    #endif
    #ifdef CMOCHC_COLLABORATION__MUTATE_NONE
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
    instance.weights = (float**)(malloc(sizeof(float*) * input.thread_count));

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
        instance.count_historic_weights = (int*)(malloc(sizeof(int) * input.thread_count));
        for (int t = 0; t < instance.input->thread_count; t++) {
            instance.count_historic_weights[t] = 0;
        }
    #endif

    /* Estado de la población de cada hilo. */
    instance.population = (struct solution**)(malloc(sizeof(struct solution*) * input.thread_count));
    instance.sorted_population = (int**)(malloc(sizeof(int*) * input.thread_count));

    /* Sync */
    if (pthread_barrier_init(&(instance.sync_barrier), NULL, input.thread_count + 1))
    {
        fprintf(stderr, "[ERROR] could not create a sync barrier.\n");
        exit(EXIT_FAILURE);
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

    /* Estado de la población elite global */
    instance.global_elite_pop = (struct solution*)(malloc(sizeof(struct solution) * (input.thread_count * CMOCHC_LOCAL__BEST_SOLS_KEPT)));

    for (int i = 0; i < (input.thread_count * CMOCHC_LOCAL__BEST_SOLS_KEPT); i++) {
        create_empty_solution(&instance.global_elite_pop[i], &current_scenario, &etc, &energy);
    }

    /* Inicializo el archivador */
    archivers_aga_init(&instance.archiver, CMOCHC_ARCHIVE__MAX_SIZE, instance.global_elite_pop, (input.thread_count * CMOCHC_LOCAL__BEST_SOLS_KEPT));
}

/* Obtiene los mejores elementos de cada población */
void gather(struct cmochc &instance) {
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
        for (int s = 0; s < instance.archiver.population_size; s++) {
            if (instance.archiver.population[s].initialized == SOLUTION__IN_USE) {
                instance.count_historic_weights[instance.archiver.population_origin[s] / CMOCHC_LOCAL__BEST_SOLS_KEPT]++;
            }
        }
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
    free(instance.population);
    free(instance.sorted_population);
    free(instance.global_elite_pop);
    free(instance.rand_state);
    free(instance.threads);

    free(threads);
}

inline int distance(struct solution *s1, struct solution *s2) {
    int distance = 0;

    for (int i = 0; i < s1->etc->tasks_count; i++) {
        if (s1->task_assignment[i] != s2->task_assignment[i]) distance++;
    }

    ASSERT(distance >= 0)
    ASSERT(distance < s1->etc->tasks_count)

    return distance;
}

inline void hux(RAND_STATE &rand_state,
    struct solution *p1, struct solution *p2,
    struct solution *c1, struct solution *c2) {

    double cross_prob = CMOCHC_LOCAL__MATING_CHANCE / (double)p1->etc->tasks_count;

    double random;
    int current_task_index = 0;

    while (current_task_index < p1->etc->tasks_count) {
        random = RAND_GENERATE(rand_state);

        int mask = 0x0;
        int mask_size = 256; // 8-bit mask
        float base_step = 1.0/(double)mask_size;
        float base = base_step;

        while (random > base) {
            base += base_step;
            mask += 0x1;
        }

        int mask_index = 0x1;
        while ((mask_index < mask_size) && (current_task_index < p1->etc->tasks_count)) {
            if ((mask & 0x1) == 1) {
                random = RAND_GENERATE(rand_state);

                if (random < cross_prob) {
                    // Si la máscara vale 1 copio las asignaciones cruzadas de la tarea
                    c1->task_assignment[current_task_index] = p2->task_assignment[current_task_index];
                    c2->task_assignment[current_task_index] = p1->task_assignment[current_task_index];
                } else {
                    // Si la máscara vale 0 copio las asignaciones derecho de la tarea
                    c1->task_assignment[current_task_index] = p1->task_assignment[current_task_index];
                    c2->task_assignment[current_task_index] = p2->task_assignment[current_task_index];
                }
            } else {
                // Si la máscara vale 0 copio las asignaciones derecho de la tarea
                c1->task_assignment[current_task_index] = p1->task_assignment[current_task_index];
                c2->task_assignment[current_task_index] = p2->task_assignment[current_task_index];
            }

            // Desplazo la máscara hacia la derecha
            mask = mask >> 1;
            mask_index = mask_index << 1;
            current_task_index++;
        }
    }

    c1->initialized = SOLUTION__IN_USE;
    c2->initialized = SOLUTION__IN_USE;

    refresh_solution(c1);
    refresh_solution(c2);
}

inline void mutate(RAND_STATE &rand_state, struct solution *seed, struct solution *mutation) {
    int current_task_index = 0;
    int tasks_count = seed->etc->tasks_count;
    int machines_count = seed->etc->machines_count;

    double mut_prob = CMOCHC_LOCAL__MUTATE_CHANCE / (double)tasks_count;

    while (current_task_index < tasks_count) {
        double random;
        random = RAND_GENERATE(rand_state);

        int mask = 0x0;
        int mask_size = 256; // 8-bit mask
        float base_step = 1.0/(double)mask_size;
        float base = base_step;

        while (random > base) {
            base += base_step;
            mask += 0x1;
        }

        int destination_machine;
        int mask_index = 0x1;
        while ((mask_index < mask_size) && (current_task_index < tasks_count)) {
            if ((mask & 0x1) == 1) {
                random = RAND_GENERATE(rand_state);

                if (random < mut_prob) {
                    random = RAND_GENERATE(rand_state);
                    destination_machine = (int)(floor(machines_count * random));

                    ASSERT(destination_machine >= 0)
                    ASSERT(destination_machine < machines_count)

                    // Si la máscara vale 1 copio reubico aleariamente la tarea
                    mutation->task_assignment[current_task_index] = destination_machine;
                } else {
                    // Si la máscara vale 0 copio las asignaciones derecho de la tarea
                    mutation->task_assignment[current_task_index] = seed->task_assignment[current_task_index];
                }
            } else {
                // Si la máscara vale 0 copio las asignaciones derecho de la tarea
                mutation->task_assignment[current_task_index] = seed->task_assignment[current_task_index];
            }

            // Desplazo la máscara hacia la derecha
            mask = mask >> 1;
            mask_index = mask_index << 1;
            current_task_index++;
        }
    }

    refresh_solution(mutation);
}

inline float fitness(struct solution *population, float *fitness_population, float *weights,
    float makespan_zenith_value, float energy_zenith_value,
    float makespan_nadir_value, float energy_nadir_value,
    int solution_index) {

    #ifdef CMOCHC_LOCAL__Z_FITNESS_NORM
        if (isnan(fitness_population[solution_index])) {
            fitness_population[solution_index] =
                ((population[solution_index].makespan/makespan_zenith_value) * weights[0]) +
                ((population[solution_index].energy_consumption/energy_zenith_value) * weights[1]);
        }
    #endif
    #ifdef CMOCHC_LOCAL__ZN_FITNESS_NORM
        if (isnan(fitness_population[solution_index])) {
            fitness_population[solution_index] =
                (((population[solution_index].makespan - makespan_zenith_value) /
                    (makespan_nadir_value - makespan_zenith_value)) * weights[0]) +
                (((population[solution_index].energy_consumption - energy_zenith_value) /
                    (energy_nadir_value - energy_zenith_value)) * weights[1]);
        }
    #endif

    return fitness_population[solution_index];
}

inline void merge_sort(struct solution *population, float *weights,
    float makespan_zenith_value, float energy_zenith_value,
    float makespan_nadir_value, float energy_nadir_value,
    int *sorted_population, float *fitness_population, int population_size);

/* Logica de los esclavos */
void* slave_thread(void *data) {
    struct cmochc_thread *t_data = (struct cmochc_thread*)data;
    struct cmochc *instance = t_data->data;

    int thread_id = t_data->thread_id;

    struct params *input = instance->input;
    struct scenario *current_scenario = instance->current_scenario;
    struct etc_matrix *etc = instance->etc;
    struct energy_matrix *energy = instance->energy;

    struct solution *arhiver_pop = instance->archiver.population;

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
    RAND_INIT(thread_id,rand_state[thread_id]);
    double random;

    /* *********************************************************************************************
     * Inicializo los pesos.
     * *********************************************************************************************/

    /* Inicializo el peso asignado a este thread */
    instance->weights[thread_id] = (float*)(malloc(sizeof(float) * 2));
    float *weights = instance->weights[thread_id];

    float thread_weight_step = 0.0;
    if (input->thread_count > 1) {
        thread_weight_step = 1.0 / (float)(input->thread_count-1);

        weights[0] = (float)thread_id * thread_weight_step;
        weights[1] = 1 - weights[0];
    } else {
        weights[0] = 0.5;
        weights[1] = 0.5;
    }

    #ifdef DEBUG_1
        fprintf(stderr, "[DEBUG] Thread %d, weight (%f,%f)\n", thread_id, weights[0], weights[1]);
    #endif

    ASSERT(weights[0] >= 0)
    ASSERT(weights[0] <= 1)
    ASSERT(weights[1] >= 0)
    ASSERT(weights[1] <= 1)

    /* Busco los threads mas cercanos */
    int *n_closest_threads = (int*)malloc(sizeof(int) * CMOCHC_COLLABORATION__MOEAD_NEIGH_SIZE);
    for (int n = 0; n < CMOCHC_COLLABORATION__MOEAD_NEIGH_SIZE; n++) n_closest_threads[n] = -1;

    float current_distance = thread_weight_step;
    float upper_bound, lower_bound;
    int upper_neigh, lower_neigh;
    int next_neigh = 0;

    if (current_distance > 0) {
        while ((current_distance <= 1.0) && (next_neigh < CMOCHC_COLLABORATION__MOEAD_NEIGH_SIZE)) {
            upper_bound = weights[0] + current_distance;
            if ((upper_bound >= 0.0)&&(upper_bound <= 1.0)) {
                upper_neigh = upper_bound / thread_weight_step;
                n_closest_threads[next_neigh] = upper_neigh;
                next_neigh++;
            }

            lower_bound = weights[0] - current_distance;
            if ((lower_bound >= 0.0)&&(lower_bound <= 1.0)&&(next_neigh < CMOCHC_COLLABORATION__MOEAD_NEIGH_SIZE)) {
                lower_neigh = lower_bound / thread_weight_step;
                n_closest_threads[next_neigh] = lower_neigh;
                next_neigh++;
            }

            current_distance += thread_weight_step;
        }
    }

    #ifdef DEBUG_1
        //if (thread_id == 1) {
            fprintf(stderr, "[DEBUG] Thread %d, closest neighbours:\n", thread_id);
            float w0,w1;
            for (int n = 0; n < CMOCHC_COLLABORATION__MOEAD_NEIGH_SIZE; n++) {
                w0 = (float)n_closest_threads[n] * thread_weight_step;
                w1 = 1 - w0;

                fprintf(stderr, "<%d> %d (%f,%f)\n", thread_id, n_closest_threads[n], w0, w1);
            }
            fprintf(stderr, "\n");
        //}
    #endif

    /* *********************************************************************************************
     * Inicializo la población.
     * *********************************************************************************************/

    /* Inicializo la población de padres y limpio la de hijos */
    int max_pop_sols = 2 * input->population_size;

    /* Poblacion de cada esclavo */
    instance->population[thread_id] = (struct solution*)(malloc(sizeof(struct solution) * max_pop_sols));
    struct solution *population = instance->population[thread_id];

    instance->sorted_population[thread_id] = (int*)(malloc(sizeof(int) * max_pop_sols));
    int *sorted_population = instance->sorted_population[thread_id];

    float *fitness_population;
    fitness_population = (float*)(malloc(sizeof(float) * max_pop_sols));

    int makespan_utopia_index, energy_utopia_index;
    float makespan_utopia_value, energy_utopia_value;
    float makespan_nadir_value, energy_nadir_value;

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
        fitness(population, fitness_population, weights,
            makespan_utopia_value, energy_utopia_value, makespan_nadir_value, energy_nadir_value, i);

        #ifdef DEBUG_3
            fprintf(stderr, "[DEBUG] Thread %d, solution=%d makespan=%.2f energy=%.2f fitness=%.2f\n",
                thread_id, i, population[i].makespan, population[i].energy_consumption,
                    fitness(population, fitness_population, weights,
                        makespan_utopia_value, energy_utopia_value,
                        makespan_nadir_value, energy_nadir_value, i));
        #endif
    }

    merge_sort(population, weights, makespan_utopia_value, energy_utopia_value,
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

            float d;
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

                        fitness(population, fitness_population, weights,
                            makespan_utopia_value, energy_utopia_value,
                            makespan_nadir_value, energy_nadir_value, c1_idx);
                        fitness(population, fitness_population, weights,
                            makespan_utopia_value, energy_utopia_value,
                            makespan_nadir_value, energy_nadir_value, c2_idx);

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

                float best_parent;
                best_parent = fitness(population, fitness_population, weights,
                    makespan_utopia_value, energy_utopia_value, makespan_nadir_value,
                    energy_nadir_value, sorted_population[0]);

                float worst_parent;
                worst_parent = fitness(population, fitness_population, weights,
                    makespan_utopia_value, energy_utopia_value, makespan_nadir_value,
                    energy_nadir_value, sorted_population[input->population_size-1]);

                merge_sort(population, weights, makespan_utopia_value, energy_utopia_value,
                    makespan_nadir_value, energy_nadir_value, sorted_population,
                    fitness_population, max_pop_sols);

                if (worst_parent > fitness(population, fitness_population, weights,
                    makespan_utopia_value, energy_utopia_value, makespan_nadir_value,
                    energy_nadir_value, sorted_population[input->population_size-1])) {

                    #ifdef DEBUG_1
                        count_at_least_one_children_inserted[thread_id]++;
                    #endif
                } else {
                    threshold -= threshold_step;
                }

                if (best_parent > fitness(population, fitness_population, weights,
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
                    double pre_mut_fitness;
                #endif

                #ifdef DEBUG_3
                    fprintf(stderr, "[DEBUG] Cataclysm!\n");
                #endif

                int aux_index;

                for (int i = CMOCHC_LOCAL__BEST_SOLS_KEPT; i < max_pop_sols; i++) { /* No muto las mejores soluciones */
                    if (population[sorted_population[i]].initialized == SOLUTION__IN_USE) {
                        #ifdef DEBUG_1
                            count_cataclysm[thread_id]++;
                            pre_mut_fitness = fitness(population, fitness_population, weights, makespan_utopia_value,
                                energy_utopia_value, makespan_nadir_value, energy_nadir_value, sorted_population[i]);
                        #endif

                        aux_index = RAND_GENERATE(rand_state[thread_id]) * CMOCHC_LOCAL__BEST_SOLS_KEPT;

                        mutate(rand_state[thread_id],
                            &population[sorted_population[aux_index]],
                            &population[sorted_population[i]]);

                        fitness_population[sorted_population[i]] = NAN;
                        fitness(population, fitness_population, weights, makespan_utopia_value, energy_utopia_value,
                            makespan_nadir_value, energy_nadir_value, sorted_population[i]);

                        #ifdef DEBUG_1
                            if (fitness_population[sorted_population[i]] < pre_mut_fitness) {
                                count_improved_mutation[thread_id]++;
                            }
                        #endif
                    }
                }

                /* Re-sort de population */
                merge_sort(population, weights, makespan_utopia_value, energy_utopia_value,
                    makespan_nadir_value, energy_nadir_value, sorted_population,
                    fitness_population, max_pop_sols);
            }
        }

        /* *********************************************************************************************
         * Espero a que los demas esclavos terminen.
         * ********************************************************************************************* */
        rc = pthread_barrier_wait(&instance->sync_barrier);
        if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
        {
            printf("Could not wait on barrier\n");
            exit(EXIT_FAILURE);
        }

        /* Copio mis mejores soluciones a la población de intercambio principal */
        for (int i = 0; i < CMOCHC_LOCAL__BEST_SOLS_KEPT; i++) {
            clone_solution(&instance->global_elite_pop[thread_id * CMOCHC_LOCAL__BEST_SOLS_KEPT + i], &population[sorted_population[i]]);
        }

        #ifdef CMOCHC_SYNC
            /* Espero a que el maestro ejecute la operación de gather. */
            rc = pthread_barrier_wait(&instance->sync_barrier);
            if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
            {
                printf("Could not wait on barrier\n");
                exit(EXIT_FAILURE);
            }
        #endif

        /* *********************************************************************************************
         * Migro soluciones desde poblaciones elite vecinas
         * ********************************************************************************************* */
        int neigh_pop, migrated, source_index, next_solution;
        next_solution = CMOCHC_LOCAL__BEST_SOLS_KEPT;
        
        while (next_solution < max_pop_sols) {
            migrated = 0;
            neigh_pop = next_solution - CMOCHC_LOCAL__BEST_SOLS_KEPT;

            #ifndef CMOCHC_COLLABORATION__MIGRATION_NONE
                count_migrations[thread_id]++;
            
                if (neigh_pop < CMOCHC_COLLABORATION__MOEAD_NEIGH_SIZE) {
                    if (n_closest_threads[neigh_pop] != -1) {
                        source_index = 0;
                        #ifdef CMOCHC_COLLABORATION__MIGRATION_RANDOM_ELITE
                            source_index = RAND_GENERATE(rand_state[thread_id]) * CMOCHC_LOCAL__BEST_SOLS_KEPT;
                        #endif

                        #ifdef CMOCHC_COLLABORATION__MIGRATE_BY_COPY
                            clone_solution(&population[sorted_population[next_solution]],
                                &(instance->population[neigh_pop][instance->sorted_population[neigh_pop][source_index]]));

                            next_solution++;
                            count_solutions_migrated[thread_id]++;
                        #endif
                        #ifdef CMOCHC_COLLABORATION__MIGRATE_BY_MATE
                            if (next_solution+1 < max_pop_sols) {
                                float d = distance(&(instance->population[neigh_pop][instance->sorted_population[neigh_pop][source_index]]),
                                    &population[sorted_population[0]]);

                                if (d > threshold_max) {
                                    hux(rand_state[thread_id],
                                        &(instance->population[neigh_pop][instance->sorted_population[neigh_pop][source_index]]), &population[sorted_population[0]],
                                        &population[sorted_population[next_solution]],&population[sorted_population[next_solution+1]]);
                                }

                                next_solution += 2;
                                count_solutions_migrated[thread_id]++;
                            }
                        #endif
                        #ifdef CMOCHC_COLLABORATION__MIGRATE_BY_MUTATE
                            mutate(rand_state[thread_id],
                                &(instance->population[neigh_pop][instance->sorted_population[neigh_pop][source_index]]),
                                &population[sorted_population[next_solution]]);

                            next_solution++;
                            count_solutions_migrated[thread_id]++;
                        #endif

                        migrated = 1;
                    }
                }
            #endif

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

        /* Actualizo los puntos de normalización con el archivo global */
        for (int i = 0; i < instance->archiver.population_size; i++) {
            if (arhiver_pop[i].initialized == SOLUTION__IN_USE) {
                if (arhiver_pop[i].makespan < makespan_utopia_value) {
                    makespan_utopia_index = i;
                    makespan_utopia_value = arhiver_pop[i].makespan;

                    if (arhiver_pop[i].energy_consumption > energy_nadir_value) {
                        energy_nadir_value = arhiver_pop[i].energy_consumption;
                    }
                }
                if (arhiver_pop[i].energy_consumption < energy_utopia_value) {
                    energy_utopia_index = i;
                    energy_utopia_value = arhiver_pop[i].energy_consumption;

                    if (arhiver_pop[i].makespan > makespan_nadir_value) {
                        makespan_nadir_value = arhiver_pop[i].makespan;
                    }
                }
            }
        }

        /* *********************************************************************************************
         * Espero a que los demas esclavos terminen.
         * ********************************************************************************************* */
        rc = pthread_barrier_wait(&instance->sync_barrier);
        if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
        {
            printf("Could not wait on barrier\n");
            exit(EXIT_FAILURE);
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
            fitness(population, fitness_population, weights, makespan_utopia_value, energy_utopia_value,
                makespan_nadir_value, energy_nadir_value, i);
        }

        /* *********************************************************************************************
         * Re-sort de population
         *********************************************************************************************** */
        merge_sort(population, weights, makespan_utopia_value, energy_utopia_value,
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

    // ================================================================
    // Finalizo el thread.
    // ================================================================
    for (int i = 0; i < max_pop_sols; i++) {
        free_solution(&(population[i]));
    }

    free(population);
    free(sorted_population);
    free(fitness_population);

    return 0;
}

inline void merge_sort(struct solution *population, float *weights,
    float makespan_zenith_value, float energy_zenith_value,
    float makespan_nadir_value, float energy_nadir_value,
    int *sorted_population, float *fitness_population,
    int population_size) {

    int increment, l, l_max, r, r_max, current, i;
    int *tmp;

    increment = 1;
    tmp = (int*)malloc(sizeof(int) * population_size);

    float fitness_r, fitness_l;

    while (increment < population_size) {
        l = 0;
        r = increment;
        l_max = r - 1;
        r_max = (l_max + increment < population_size) ? l_max + increment : population_size - 1;

        current = 0;

        while (current < population_size) {
            while (l <= l_max && r <= r_max) {
                fitness_r = fitness(population, fitness_population, weights,
                    makespan_zenith_value, energy_zenith_value,
                    makespan_nadir_value, energy_nadir_value,
                    sorted_population[r]);
                fitness_l = fitness(population, fitness_population, weights,
                    makespan_zenith_value, energy_zenith_value,
                    makespan_nadir_value, energy_nadir_value,
                    sorted_population[l]);

                /*fitness_r = fitness_population[sorted_population[r]];
                fitness_l = fitness_population[sorted_population[l]];*/

                if (!isnan(fitness_r) && !isnan(fitness_l)) {
                    if (fitness_r < fitness_l) {
                        tmp[current] = sorted_population[r++];
                    } else {
                        tmp[current] = sorted_population[l++];
                    }
                } else if (!isnan(fitness_r) && isnan(fitness_l)) {
                    tmp[current] = sorted_population[r++];
                } else if (isnan(fitness_r) && !isnan(fitness_l)) {
                    tmp[current] = sorted_population[l++];
                } else {
                    /* Ambos son NAN, no importa */
                    tmp[current] = sorted_population[l++];
                }

                current++;
            }

            while (r <= r_max) tmp[current++] = sorted_population[r++];
            while (l <= l_max) tmp[current++] = sorted_population[l++];

            l = r;
            r += increment;
            l_max = r - 1;
            r_max = (l_max + increment < population_size) ? l_max + increment : population_size - 1;
        }

        increment *= 2;

        for (i = 0; i < population_size; i++) {
            sorted_population[i] = tmp[i];
        }
    }

    free(tmp);
}
