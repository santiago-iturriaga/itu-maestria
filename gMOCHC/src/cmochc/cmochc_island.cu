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

//#define LOCAL_ITERATION_COUNT   1
#define LOCAL_ITERATION_COUNT   50
//#define LOCAL_ITERATION_COUNT   100
//#define LOCAL_ITERATION_COUNT   2000

#define BEST_SOLS_KEPT  3

#define CROSS_MAX_THRESHOLD_DIVISOR     4
/* Aprox. one cataclysm every 5 local iterations without change */
#define CROSS_THRESHOLD_STEP_DIVISOR    10

#define MAX_GLOBAL_POP  50

/* Only hux_custom */
#define CROSSOVER_FLIP_PROB     0.25

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
    int *generations_no_children_born;
    int *generations_no_children_inserted;
    int *generations_at_least_one_children_inserted;
    int *generations_improved_sols;
    int *generations_cataclysm_count;
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
        fprintf(stdout, "%d\n", instance.archiver.population_count);
        for (int i = 0; i < MAX_GLOBAL_POP; i++) {
            if (instance.archiver.population[i].initialized == 1) {
                for (int task_id = 0; task_id < instance.etc->tasks_count; task_id++) {
                    fprintf(stdout, "%d\n", instance.archiver.population[i].task_assignment[task_id]);
                }
            }
        }
    #endif

    fprintf(stderr, "[DEBUG] Elite archive solutions [makespan energy]\n");

    for (int i = 0; i < MAX_GLOBAL_POP; i++) {
        if (instance.archiver.population[i].initialized == 1) {
            fprintf(stderr, "%f %f\n",
                instance.archiver.population[i].makespan, 
                instance.archiver.population[i].energy_consumption);
        }
    }
    fprintf(stderr, "> total solutions: %d\n", instance.archiver.population_count);

    #ifdef DEBUG_1
        fprintf(stderr, "[INFO] == Statistics ==========================================\n");
        for (int t = 0; t < instance.input->thread_count; t++)
            fprintf(stderr, "       [thread %d] NO CHILDREN BORN COUNT     : %d\n", t, instance.generations_no_children_born[t]);
        for (int t = 0; t < instance.input->thread_count; t++)
            fprintf(stderr, "       [thread %d] NO CHILDREN INSERTED COUNT : %d\n", t, instance.generations_no_children_inserted[t]);
        for (int t = 0; t < instance.input->thread_count; t++)
            fprintf(stderr, "       [thread %d] AT LEAST ONE CHILDREN INS. : %d\n", t, instance.generations_at_least_one_children_inserted[t]);
        for (int t = 0; t < instance.input->thread_count; t++)
            fprintf(stderr, "       [thread %d] IMPROVED SOLUTIONS COUNT   : %d\n", t, instance.generations_improved_sols[t]);
        for (int t = 0; t < instance.input->thread_count; t++)
            fprintf(stderr, "       [thread %d] CATACLYSM COUNT            : %d\n", t, instance.generations_cataclysm_count[t]);
        fprintf(stderr, "[INFO] ========================================================\n");
    #endif
}

/* Inicializa los hilos y las estructuras de datos */
void init(struct cmochc &instance, struct cmochc_thread **threads_data,
    struct params &input, struct scenario &current_scenario,
    struct etc_matrix &etc, struct energy_matrix &energy) {

    fprintf(stderr, "[INFO] == Global configuration constants ======================\n");
    fprintf(stderr, "       LOCAL ITERATION_COUNT        : %d\n", LOCAL_ITERATION_COUNT);
    fprintf(stderr, "       CROSS_MAX_THRESHOLD_DIVISOR  : %d\n", CROSS_MAX_THRESHOLD_DIVISOR);
    fprintf(stderr, "       CROSS_THRESHOLD_STEP_DIVISOR : %d\n", CROSS_THRESHOLD_STEP_DIVISOR);
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
    instance.generations_no_children_born = (int*)(malloc(sizeof(int) * input.thread_count));
    instance.generations_no_children_inserted = (int*)(malloc(sizeof(int) * input.thread_count));
    instance.generations_at_least_one_children_inserted = (int*)(malloc(sizeof(int) * input.thread_count));
    instance.generations_improved_sols = (int*)(malloc(sizeof(int) * input.thread_count));
    instance.generations_cataclysm_count = (int*)(malloc(sizeof(int) * input.thread_count));

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
    instance.global_elite_pop = (struct solution*)(malloc(sizeof(struct solution) * (input.thread_count * BEST_SOLS_KEPT)));
    
    for (int i = 0; i < (input.thread_count * BEST_SOLS_KEPT); i++) {
        create_empty_solution(&instance.global_elite_pop[i], &current_scenario, &etc, &energy);
    }
    
    /* Inicializo el archivador */
    archivers_aga_init(&instance.archiver, MAX_GLOBAL_POP, instance.global_elite_pop, (input.thread_count * BEST_SOLS_KEPT));
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
}

/* Libera los recursos pedidos y finaliza la ejecución */
void finalize(struct cmochc &instance, struct cmochc_thread *threads) {
    archivers_aga_free(&instance.archiver);
    pthread_barrier_destroy(&(instance.sync_barrier));
    
    free(instance.generations_cataclysm_count);
    free(instance.generations_improved_sols);
    free(instance.generations_no_children_born);
    free(instance.generations_no_children_inserted);
    free(instance.generations_at_least_one_children_inserted);
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

inline void hux_custom(RAND_STATE &rand_state,
    struct solution *p1, struct solution *p2,
    struct solution *c1, struct solution *c2) {

    double random;
    for (int task_index = 0; task_index < p1->etc->tasks_count; task_index++) {
        random = RAND_GENERATE(rand_state);

        if (random <= CROSSOVER_FLIP_PROB) {
            /* Si la máscara vale 1 copio las asignaciones cruzadas de la tarea */
            c1->task_assignment[task_index] = p2->task_assignment[task_index];
            c2->task_assignment[task_index] = p1->task_assignment[task_index];
        } else {
            /* Si la máscara vale 0 copio las asignaciones derecho de la tarea */
            c1->task_assignment[task_index] = p1->task_assignment[task_index];
            c2->task_assignment[task_index] = p2->task_assignment[task_index];
        }
    }

    refresh_solution(c1);
    refresh_solution(c2);
}

inline void hux(RAND_STATE &rand_state,
    struct solution *p1, struct solution *p2,
    struct solution *c1, struct solution *c2) {

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
                
                if (random < 0.1) {
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

    c1->initialized = 1;
    c2->initialized = 1;

    refresh_solution(c1);
    refresh_solution(c2);
}

inline void mutate(RAND_STATE &rand_state, struct solution *seed, struct solution *mutation) {
    int current_task_index = 0;
    int tasks_count = seed->etc->tasks_count;
    int machines_count = seed->etc->machines_count;

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
                
                if (random < 0.1) {
                    random = RAND_GENERATE(rand_state);
                    destination_machine = (int)(floor(machines_count * random));
                    
                    ASSERT(destination_machine >= 0)
                    ASSERT(destination_machine < machines_count)
                    
                    // Si la máscara vale 1 copio reubico aleariamente la tarea
                    mutation->task_assignment[current_task_index] = destination_machine;
                    
                    //#ifdef DEBUG_3
                    //    fprintf(stderr, "task=%d>>machine=%d, ", current_task_index, destination_machine);
                    //#endif
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
    float makespan_norm_value, float energy_norm_value, int index) {
        
    if (isnan(fitness_population[index])) {
        fitness_population[index] = ((population[index].makespan/makespan_norm_value) * weights[0]) +
            ((population[index].energy_consumption/energy_norm_value) * weights[1]);
    }

    return fitness_population[index];
}

inline void merge_sort(struct solution *population, float *weights, 
    float makespan_norm_value, float energy_norm_value, int *sorted_population, 
    float *fitness_population, int population_size);

/* Logica de los esclavos */
void* slave_thread(void *data) {
    struct cmochc_thread *t_data = (struct cmochc_thread*)data;
    struct cmochc *instance = t_data->data;

    int thread_id = t_data->thread_id;

    struct params *input = instance->input;
    struct scenario *current_scenario = instance->current_scenario;
    struct etc_matrix *etc = instance->etc;
    struct energy_matrix *energy = instance->energy;

    RAND_STATE *rand_state = instance->rand_state;

    int *generations_no_children_born = instance->generations_no_children_born;
    int *generations_no_children_inserted = instance->generations_no_children_inserted;
    int *generations_at_least_one_children_inserted = instance->generations_at_least_one_children_inserted;
    int *generations_improved_sols = instance->generations_improved_sols;
    int *generations_cataclysm_count = instance->generations_cataclysm_count;

    float makespan_norm_value, energy_norm_value;

    // ================================================================
    // Inicializo el thread.
    // ================================================================

    generations_no_children_born[thread_id] = 0;
    generations_no_children_inserted[thread_id] = 0;
    generations_at_least_one_children_inserted[thread_id] = 0;
    generations_improved_sols[thread_id] = 0;
    generations_cataclysm_count[thread_id] = 0;

    /* Inicialización del estado del generador aleatorio */
    RAND_INIT(thread_id,rand_state[thread_id]);
    double random;

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
    int *n_closest_threads = (int*)malloc(sizeof(int) * input->thread_count);
    for (int n = 0; n < input->thread_count; n++) n_closest_threads[n] = -1;
    
    float current_distance = thread_weight_step;
    float upper_bound, lower_bound;
    int upper_neigh, lower_neigh;
    int next_neigh = 0;
    
    if (current_distance > 0) {
        while ((current_distance <= 1.0) && (next_neigh < input->thread_count)) {
            upper_bound = weights[0] + current_distance;
            if ((upper_bound >= 0.0)&&(upper_bound <= 1.0)) {
                upper_neigh = upper_bound / thread_weight_step;
                n_closest_threads[next_neigh] = upper_neigh;
                next_neigh++;
            }

            lower_bound = weights[0] - current_distance;
            if ((lower_bound >= 0.0)&&(lower_bound <= 1.0)&&(next_neigh < input->thread_count)) {
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
            for (int n = 0; n < input->thread_count; n++) {
                w0 = (float)n_closest_threads[n] * thread_weight_step;
                w1 = 1 - w0;
                
                fprintf(stderr, "<%d> %d (%f,%f)\n", thread_id, n_closest_threads[n], w0, w1);
            }
            fprintf(stderr, "\n");
        //}
    #endif

    /* Inicializo la población de padres y limpio la de hijos */
    int max_pop_sols = 2 * input->population_size;

    /* Poblacion de cada esclavo */
    instance->population[thread_id] = (struct solution*)(malloc(sizeof(struct solution) * max_pop_sols));
    struct solution *population = instance->population[thread_id];

    instance->sorted_population[thread_id] = (int*)(malloc(sizeof(int) * max_pop_sols));
    int *sorted_population = instance->sorted_population[thread_id];

    float *fitness_population;
    fitness_population = (float*)(malloc(sizeof(float) * max_pop_sols));

    for (int i = 0; i < max_pop_sols; i++) {
        // Random init.
        create_empty_solution(&(population[i]),current_scenario,etc,energy);

        if (i > 0) {
            random = RAND_GENERATE(rand_state[thread_id]);
            int starting_pos;
            starting_pos = (int)(floor(etc->tasks_count * random));

            #ifdef DEBUG_3
                fprintf(stderr, "[DEBUG] Thread %d, inicializando solution %d, starting %d, direction %d...\n",
                    thread_id, i, starting_pos, i & 0x1);
            #endif

            compute_mct_random(&(population[i]), starting_pos, i & 0x1);
        } else {
            compute_mct(&(population[i]));
            
            makespan_norm_value = population[i].makespan;
            energy_norm_value = population[i].energy_consumption;
            
            #ifdef DEBUG_3
                fprintf(stderr, "[DEBUG] Thread %d, makespan_norm_value=%f energy_norm_value=%f\n",
                    thread_id, makespan_norm_value, energy_norm_value);
            #endif
        }
        
        sorted_population[i] = i;
        fitness_population[i] = NAN;
        fitness(population, fitness_population, weights, 
            makespan_norm_value, energy_norm_value, i);

        #ifdef DEBUG_3
            fprintf(stderr, "[DEBUG] Thread %d, inicializado solution %d, fitness %f\n",
                thread_id, i, fitness(population, fitness_population, weights, 
                    makespan_norm_value, energy_norm_value, i));
        #endif
    }

    // ================================================================
    // Main iteration
    // ================================================================
    int next_avail_children;
    int max_children = input->population_size / 2;

    int max_distance = etc->tasks_count;

    int threshold_max = max_distance / CROSS_MAX_THRESHOLD_DIVISOR;
    int threshold_step = threshold_max / CROSS_THRESHOLD_STEP_DIVISOR;
    if (threshold_step == 0) threshold_step = 1;
    int threshold = threshold_max;

    #ifdef DEBUG_1
        fprintf(stderr, "[DEBUG] Threshold Max %d.\n", threshold_max);
        fprintf(stderr, "[DEBUG] Threshold Step %d.\n", threshold_step);
    #endif

    int rc;

    while (instance->stopping_condition == 0) {
        for (int iteracion = 0; iteracion < LOCAL_ITERATION_COUNT; iteracion++) {
            #ifdef DEBUG_3
                fprintf(stderr, "[DEBUG] Iteration %d.\n", iteracion);
            #endif

            // =======================================================
            // Mating
            // =======================================================
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

                    /*
                    #ifdef DEBUG_3
                        fprintf(stderr, "[DEBUG] Selected parents %d(%d) and %d(%d) [%d]\n", p1_idx, p1_rand, p2_idx, p2_rand, input->population_size);
                    #endif
                    * */

                    // Chequeo la distancia entre padres
                    d = distance(&population[p1_idx],&population[p2_idx]);

                    if (d > threshold) {
                        // Aplico HUX y creo dos hijos
                        c1_idx = sorted_population[next_avail_children];
                        c2_idx = sorted_population[next_avail_children+1];

                        hux(rand_state[thread_id],
                            &population[p1_idx],&population[p2_idx],
                            &population[c1_idx],&population[c2_idx]);

                        fitness_population[c1_idx] = NAN;
                        fitness_population[c2_idx] = NAN;
            
                        fitness(population, fitness_population, weights, 
                            makespan_norm_value, energy_norm_value, c1_idx);
                        fitness(population, fitness_population, weights, 
                            makespan_norm_value, energy_norm_value, c2_idx);

                        next_avail_children += 2;
                    }
                }
            }

            if (next_avail_children > input->population_size) {
                #ifdef DEBUG_3
                    fprintf(stderr, "[DEBUG] %d children born.\n", next_avail_children - input->population_size);
                #endif

                // =======================================================
                // Sort parent+children population
                // =======================================================

                float best_parent;
                best_parent = fitness(population, fitness_population, weights, 
                    makespan_norm_value, energy_norm_value, sorted_population[0]);

                float worst_parent;
                worst_parent = fitness(population, fitness_population, weights, 
                    makespan_norm_value, energy_norm_value, sorted_population[input->population_size-1]);

                merge_sort(population, weights, makespan_norm_value, energy_norm_value, 
                    sorted_population, fitness_population, max_pop_sols);

                #ifdef DEBUG_3
                    fprintf(stderr, "[DEBUG] Post-sorted population\n");
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

                if (worst_parent > fitness(population, fitness_population, weights, 
                    makespan_norm_value, energy_norm_value, sorted_population[input->population_size-1])) {
                        
                    #ifdef DEBUG_1
                        generations_at_least_one_children_inserted[thread_id]++;
                    #endif

                    #ifdef DEBUG_3
                        fprintf(stderr, "[DEBUG] At least one children inserted.\n");
                    #endif
                } else {
                    #ifdef DEBUG_1
                        generations_no_children_inserted[thread_id]++;
                    #endif

                    threshold -= threshold_step;

                    #ifdef DEBUG_3
                        fprintf(stderr, "[DEBUG] No children inserted into the population.\n");
                    #endif
                }
                
                if (best_parent > fitness(population, fitness_population, weights, 
                    makespan_norm_value, energy_norm_value, sorted_population[0])) {
                        
                    #ifdef DEBUG_1
                        generations_improved_sols[thread_id]++;
                    #endif

                    #ifdef DEBUG_3
                        fprintf(stderr, "[DEBUG] Current best solution was improved (from %f to %f)!\n",
                            best_parent, fitness(population, fitness_population, weights, 
                                makespan_norm_value, energy_norm_value, sorted_population[0]));
                    #endif
                }
            } else {
                #ifdef DEBUG_1
                    generations_no_children_born[thread_id]++;
                #endif

                threshold -= threshold_step;

                #ifdef DEBUG_3
                    fprintf(stderr, "[DEBUG] No children born.\n");
                #endif
            }

            #ifdef DEBUG_3
                fprintf(stderr, "[DEBUG] Threshold %d\n", threshold);
            #endif

            if (threshold < 0) {
                threshold = threshold_max;

                // =======================================================
                // Cataclysm
                // =======================================================
                
                #ifdef DEBUG_1
                    generations_cataclysm_count[thread_id]++;
                #endif

                #ifdef DEBUG_3
                    fprintf(stderr, "[DEBUG] Cataclysm %d!\n", generations_cataclysm_count[thread_id]);
                #endif
               
                int solution_seed;
                for (int i = BEST_SOLS_KEPT; i < max_pop_sols; i++) { /* No muto la mejor solución */
                    if (population[sorted_population[i]].initialized == 1) {
                        solution_seed = RAND_GENERATE(rand_state[thread_id]) * BEST_SOLS_KEPT;
                        mutate(rand_state[thread_id], &population[sorted_population[solution_seed]], &population[sorted_population[i]]);
                        
                        fitness_population[sorted_population[i]] = NAN;
                        fitness(population, fitness_population, weights, makespan_norm_value, energy_norm_value, sorted_population[i]);
                    }
                }
                #ifdef DEBUG_3
                    fprintf(stderr, "\n");
                #endif
                
                /* Re-sort de population */
                merge_sort(population, weights, makespan_norm_value, energy_norm_value, 
                    sorted_population, fitness_population, max_pop_sols);

                #ifdef DEBUG_3
                    fprintf(stderr, "[DEBUG] Post-mutate population\n");
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
                
        /* ***************************************** */
        /* Espero a que los demas esclavos terminen. */
        /* ***************************************** */
        rc = pthread_barrier_wait(&instance->sync_barrier);
        if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
        {
            printf("Could not wait on barrier\n");
            exit(EXIT_FAILURE);
        }
        
        /* Copio mi mejor solución a la población temp principal */
        for (int i = 0; i < BEST_SOLS_KEPT; i++) {
            #ifdef DEBUG_3
                fprintf(stderr, "[DEBUG] Thread %d, copiando mejor sol. %d en pos %d a para gather\n", 
                    thread_id, i, sorted_population[i]);
            #endif
            clone_solution(&instance->global_elite_pop[thread_id * BEST_SOLS_KEPT + i], &population[sorted_population[i]]);
        }
        
        /* Migro soluciones desde poblaciones elite vecinas */
        int neigh_pop, migrated;
        for (int n = BEST_SOLS_KEPT; n < max_pop_sols; n++) {
            migrated = 0;
            neigh_pop = n-BEST_SOLS_KEPT;
            
            /*if (neigh_pop < input->thread_count) {
                if (n_closest_threads[neigh_pop] != -1) {
                    #ifdef DEBUG_3
                        fprintf(stderr, "[DEBUG] Thread %d, migrando solución %d desde %d (to pos %d)\n", 
                            thread_id, instance->sorted_population[neigh_pop][1], neigh_pop, sorted_population[n]);
                    #endif
                    
                    clone_solution(&population[sorted_population[n]], 
                        &(instance->population[neigh_pop][instance->sorted_population[neigh_pop][1]]));
                        
                    migrated = 1;
                }
            } */
            
            if (migrated == 0) {
                #ifdef DEBUG_3
                    fprintf(stderr, "[DEBUG] Thread %d, muto la sol %d usando la sol %d\n", 
                        thread_id, sorted_population[n], sorted_population[0]);
                #endif
                
                mutate(rand_state[thread_id], &population[sorted_population[0]], &population[sorted_population[n]]);
            }
            
            fitness_population[sorted_population[n]] = NAN;
            fitness(population, fitness_population, weights, makespan_norm_value, energy_norm_value, sorted_population[n]);
        }
        
        /* ***************************************** */
        /* Espero a que los demas esclavos terminen. */
        /* ***************************************** */
        rc = pthread_barrier_wait(&instance->sync_barrier);
        if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
        {
            printf("Could not wait on barrier\n");
            exit(EXIT_FAILURE);
        }

        // Muto la mejor solución que tenía.
        //for (int i = 0; i < BEST_SOLS_KEPT; i++) {
        for (int i = 0; i < 1; i++) {
            mutate(rand_state[thread_id], &population[sorted_population[i]], &population[sorted_population[i]]);
            fitness_population[sorted_population[i]] = NAN;
            fitness(population, fitness_population, weights, makespan_norm_value, energy_norm_value, sorted_population[i]);
        }
                
        /* Re-sort de population */
        merge_sort(population, weights, makespan_norm_value, energy_norm_value, 
            sorted_population, fitness_population, max_pop_sols);

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
    float makespan_norm_value, float energy_norm_value, int *sorted_population, 
    float *fitness_population, int population_size) {
        
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
                    makespan_norm_value, energy_norm_value, sorted_population[r]);
                fitness_l = fitness(population, fitness_population, weights, 
                    makespan_norm_value, energy_norm_value, sorted_population[l]);
                
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
