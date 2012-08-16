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

#define LOCAL_ITERATION_COUNT   5
#define GLOBAL_ELITE_POP_SIZE   3

struct cmochc {
    struct params *input;
    struct scenario *current_scenario;
    struct etc_matrix *etc;
    struct energy_matrix *energy;

    /* Coleccion de esclavos */
    pthread_t *threads;

    /* Poblacion de cada esclavo */
    struct solution **population;
    /* Solucion elite local a cada esclavo */
    struct solution *local_elite_sol;
    /* Poblacion elite global mantenida por el master */
    struct solution global_elite_pop[GLOBAL_ELITE_POP_SIZE];
    
    float **weights;

    /* Poblacion de cada esclavo */
    RAND_STATE *rand_state;
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

/* Evoluciona las poblaciones */
void evolve(struct cmochc &instance);

/* Obtiene los mejores elementos de cada población */
void gather(struct cmochc &instance);

/* Migra los mejores elementos a poblaciones vecinas */
void migrate(struct cmochc &instance);

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

    for (int iteracion = 0; iteracion < input.max_iterations; iteracion++) {
        // Timming -----------------------------------------------------
        TIMMING_START(ts_evolve);
        // Timming -----------------------------------------------------
        #if defined(DEBUG_3)
            fprintf(stderr, "[DEBUG] CPU CHC (islands): evolve\n");
        #endif

        evolve(instance);

        // Timming -----------------------------------------------------
        TIMMING_END(">> cmochc_evolve", ts_evolve);
        // Timming -----------------------------------------------------

        // Timming -----------------------------------------------------
        TIMMING_START(ts_gather);
        // Timming -----------------------------------------------------
        #if defined(DEBUG_3)
            fprintf(stderr, "[DEBUG] CPU CHC (islands): gather\n");
        #endif

        gather(instance);

        // Timming -----------------------------------------------------
        TIMMING_END(">> cmochc_gather", ts_gather);
        // Timming -----------------------------------------------------


        // Timming -----------------------------------------------------
        TIMMING_START(ts_migrate);
        // Timming -----------------------------------------------------
        #if defined(DEBUG_3)
            fprintf(stderr, "[DEBUG] CPU CHC (islands): migrate\n");
        #endif

        migrate(instance);

        // Timming -----------------------------------------------------
        TIMMING_END(">> cmochc_migrate", ts_migrate);
        // Timming -----------------------------------------------------
    }

    // Bloqueo la ejecucion hasta que terminen todos los hilos.
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

    // Libero la memoria.
    #if defined(DEBUG_1)
        fprintf(stderr, "[DEBUG] CPU CHC (islands): finalize\n");
    #endif
    finalize(instance, threads);
}

/* Inicializa los hilos y las estructuras de datos */
void init(struct cmochc &instance, struct cmochc_thread **threads_data,
    struct params &input, struct scenario &current_scenario,
    struct etc_matrix &etc, struct energy_matrix &energy) {

    fprintf(stderr, "[INFO] == Configuration constants =============================\n");
    fprintf(stderr, "       LOCAL ITERATION_COUNT   : %d\n", LOCAL_ITERATION_COUNT);
    fprintf(stderr, "       GLOBAL_ELITE_POP_SIZE   : %d\n", GLOBAL_ELITE_POP_SIZE);
    fprintf(stderr, "[INFO] ========================================================\n");

    // Estado relacionado con el problema.
    instance.input = &input;
    instance.current_scenario = &current_scenario;
    instance.etc = &etc;
    instance.energy = &energy;

    // Estado del generador aleatorio.
    instance.rand_state = (RAND_STATE*)(malloc(sizeof(RAND_STATE) * input.thread_count));

    // Weights
    instance.weights = (float**)(malloc(sizeof(float*) * input.thread_count));

    // Estado de la población.
    instance.population = (struct solution**)(malloc(sizeof(struct solution*) * input.thread_count));
    instance.local_elite_sol = (struct solution*)(malloc(sizeof(struct solution) * input.thread_count));

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
            fprintf(stderr, "Could not create slave thread %d\n", i);
            exit(EXIT_FAILURE);
        }
    }

    for (int i = 0; i < GLOBAL_ELITE_POP_SIZE; i++) {
        instance.global_elite_pop[i].initialized = 0;
    }
}

/* Evoluciona las poblaciones */
void evolve(struct cmochc &instance) {

}

/* Obtiene los mejores elementos de cada población */
void gather(struct cmochc &instance) {

}

/* Migra los mejores elementos a poblaciones vecinas */
void migrate(struct cmochc &instance) {

}

/* Libera los recursos pedidos y finaliza la ejecución */
void finalize(struct cmochc &instance, struct cmochc_thread *threads) {
    free(instance.weights);
    free(instance.population);
    free(instance.local_elite_sol);
    free(instance.rand_state);
    free(instance.threads);
    free(threads);
}

int distance(struct solution *s1, struct solution *s2) {
    int distance = 0;

    for (int i = 0; i < s1->etc->tasks_count; i++) {
        if (s1->task_assignment[i] != s2->task_assignment[i]) distance++;
    }

    ASSERT(distance >= 0)
    ASSERT(distance < s1->etc->tasks_count)

    return distance;
}

void hux(RAND_STATE &rand_state,
    struct solution *p1, struct solution *p2,
    struct solution *c1, struct solution *c2) {

    int current_task_index = 0;

    while (current_task_index < p1->etc->tasks_count) {
        double random;
        random = RAND_GENERATE(rand_state);

        int mask = 0x0;
        int mask_size = 256; /* 8-bit mask */
        float base_step = 1.0/(double)mask_size;
        float base = base_step;

        while (random > base) {
            base += base_step;
            mask += 0x1;
        }

        /*
        #ifdef DEBUG_3
            char *aux;
            aux = int_to_binary(mask);
            
            fprintf(stderr, "[DEBUG] Bit mask: %d | %x | %s\n", mask, mask, aux);
            
            free(aux);
        #endif
        * */

        int mask_index = 0x1;
        while ((mask_index < mask_size) && (current_task_index < p1->etc->tasks_count)) {
            if ((mask & 0x1) == 1) {
                /* Si la máscara vale 1 copio las asignaciones cruzadas de la tarea */
                c1->task_assignment[current_task_index] = p2->task_assignment[current_task_index];
                c2->task_assignment[current_task_index] = p1->task_assignment[current_task_index];
            } else {
                /* Si la máscara vale 0 copio las asignaciones derecho de la tarea */
                c1->task_assignment[current_task_index] = p1->task_assignment[current_task_index];
                c2->task_assignment[current_task_index] = p2->task_assignment[current_task_index];
            }

            /* Desplazo la máscara hacia la derecha */
            mask = mask >> 1;
            mask_index = mask_index << 1;
            current_task_index++;
        }
    }
    
    refresh_solution(c1);
    refresh_solution(c2);

    #ifdef DEBUG_3
        validate_solution(c1);
        validate_solution(c2);
    #endif
}

float fitness(struct solution *population, float *fitness_population, float *weights, int index) {
    if (isnan(fitness_population[index])) {
        fitness_population[index] = (population[index].makespan * weights[0]) + 
            (population[index].energy_consumption * weights[1]);
    }
    
    return fitness_population[index];
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

    RAND_STATE *rand_state = instance->rand_state;

    // ================================================================
    // Inicializo el thread.
    // ================================================================

    /* Inicialización del estado del generador aleatorio */
    RAND_INIT(thread_id,rand_state[thread_id]);
    double random;

    /* Inicializo el peso asignado a este thread */
    instance->weights[thread_id] = (float*)(malloc(sizeof(float) * 2));
    float *weights = instance->weights[thread_id];

    if (input->thread_count > 1) {
        float thread_weight = (float)thread_id / (float)(input->thread_count-1);
        weights[0] = thread_weight;
        weights[1] = 1 - thread_weight;
    } else {
        weights[0] = 1;
        weights[1] = 1;
    }

    #ifdef DEBUG_3
        fprintf(stderr, "[DEBUG] Thread %d, weight (%f,%f)\n", thread_id, weights[0], weights[1]);
    #endif
    
    ASSERT(weights[0] >= 0)
    ASSERT(weights[0] <= 1)
    ASSERT(weights[1] >= 0)
    ASSERT(weights[1] <= 1)

    /* Inicializo la población de padres y limpio la de hijos */
    int max_pop_sols = 2 * input->population_size;

    /* Poblacion de cada esclavo */
    instance->population[thread_id] = (struct solution*)(malloc(sizeof(struct solution) * max_pop_sols));
    struct solution *population = instance->population[thread_id];

    int *sorted_population;
    sorted_population = (int*)(malloc(sizeof(int) * max_pop_sols));

    float *fitness_population;
    fitness_population = (float*)(malloc(sizeof(float) * max_pop_sols));

    for (int i = 0; i < input->population_size; i++) {
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
        
        sorted_population[i] = i;
        fitness_population[i] = NAN;
        
        fitness(population, fitness_population, weights, i);
        
        #ifdef DEBUG_3
            fprintf(stderr, "[DEBUG] Thread %d, inicializado solution %d, fitness %f\n", 
                thread_id, i, fitness(population, fitness_population, weights, i));
        #endif
    }

    for (int i = instance->input->population_size; i < max_pop_sols; i++) {
        create_empty_solution(&(population[i]),current_scenario,etc,energy);

        sorted_population[i] = i;
        fitness_population[i] = NAN;
    }

    /* Limpio la solución elite */
    struct solution *local_elite_sol = &(instance->local_elite_sol[thread_id]);
    local_elite_sol->initialized = 0;
    
    float local_elite_fitness;
    local_elite_fitness = NAN;
    
    // ================================================================
    // .
    // ================================================================
    int next_avail_children;
    int max_children = input->population_size / 2;

    int max_distance = etc->tasks_count;
    int max_threshold = max_distance / 4;
    int threshold = max_threshold;

    // http://en.wikipedia.org/wiki/Merge_sort
    // http://en.wikipedia.org/wiki/Insertion_sort
    // http://blog.macuyiko.com/2009/01/modern-genetic-and-other-algorithms_237.html

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

                ASSERT(p1_idx != p2_idx)
                ASSERT(p1_idx >= 0)
                ASSERT(p1_idx < input->population_size)
                ASSERT(p2_idx >= 0)
                ASSERT(p2_idx < input->population_size)

                // Chequeo la distancia entre padres
                d = distance(&population[p1_idx],&population[p2_idx]);
                
                #ifdef DEBUG_3
                    fprintf(stderr, "[DEBUG] Selected parents %d and %d > distance = %f\n", p1_idx, p2_idx, d);
                #endif
                
                if (d > threshold) {
                    // Aplico HUX y creo dos hijos
                    c1_idx = sorted_population[next_avail_children];
                    c2_idx = sorted_population[next_avail_children+1];

                    hux(rand_state[thread_id],
                        &population[p1_idx],&population[p2_idx],
                        &population[c1_idx],&population[c2_idx]);

                    fitness(population, fitness_population, weights, c1_idx);
                    fitness(population, fitness_population, weights, c2_idx);

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

            // ...
        } else {
            threshold--;

            #ifdef DEBUG_3
                fprintf(stderr, "[DEBUG] No children born.\n");
            #endif
        }

        if (threshold < 0) {
            threshold = max_threshold;

            int best_sol_index;
            best_sol_index = sorted_population[0];

            if (local_elite_sol->initialized == 0) {
                // Si la solución elite no esta inicializada...
                clone_solution(local_elite_sol, &population[best_sol_index]);
                local_elite_fitness = fitness(population, fitness_population, weights, best_sol_index);
                
            } else if (local_elite_fitness > fitness(population, fitness_population, weights, best_sol_index)) {
                // O si la mejor solución de la población es mejor
                // que la solución elite...
                clone_solution(local_elite_sol, &population[best_sol_index]);
                local_elite_fitness = fitness(population, fitness_population, weights, best_sol_index);
                
            }

            // =======================================================
            // Cataclysm
            // =======================================================

            // ...
        }
        
        // =======================================================
        // Reset children population
        // =======================================================
        int sorted_i;
        for (int i = instance->input->population_size; i < max_pop_sols; i++) {
            sorted_i = sorted_population[i];
            
            population[sorted_i].initialized = 0;
            fitness_population[sorted_i] = NAN;
        }
    }

    // ================================================================
    // Finalizo el thread.
    // ================================================================
    free(instance->population);
    free(sorted_population);
    free(fitness_population);

    return 0;
}
