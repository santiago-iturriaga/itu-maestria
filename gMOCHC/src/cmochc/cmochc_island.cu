#include <pthread.h>

#include "cmochc_island.h"

#include "../config.h"
#include "../solution.h"
#include "../load_params.h"
#include "../scenario.h"
#include "../etc_matrix.h"
#include "../energy_matrix.h"
#include "../utils.h"
#include "../basic/mct.h"

#define LOCAL_ITERATION_COUNT   50
#define GLOBAL_ITERATION_COUNT  50

#define LOCAL_ELITE_POP_SIZE    1
#define GLOBAL_ELITE_POP_SIZE   50

struct cmochc {
    struct params *input;
    struct scenario *current_scenario;
    struct etc_matrix *etc;
    struct energy_matrix *energy;

    /* Coleccion de esclavos */
    pthread_t *threads;

    /* Poblacion de cada esclavo */
    struct solution **population;
    /* Poblacion elite local a cada esclavo */
    struct solution **local_elite_pop;
    /* Poblacion elite global mantenida por el master */
    struct solution global_elite_pop[GLOBAL_ELITE_POP_SIZE];
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
        fprintf(stderr, "[DEBUG] CPU CHC (islands): init");
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

    for (int iteracion = 0; iteracion < GLOBAL_ITERATION_COUNT; iteracion++) {
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
            printf("Could not join thread %d\n", i);
            exit(EXIT_FAILURE);
        }
        else
        {
            #if defined(DEBUG)
            printf("[DEBUG] thread %d <OK>\n", i);
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

    fprintf(stdout, "[INFO] == Configuration constants =============================\n");
    fprintf(stdout, "       ITERATION_COUNT         : %d\n", ITERATION_COUNT);
    fprintf(stdout, "       LOCAL_ELITE_POP_SIZE    : %d\n", LOCAL_ELITE_POP_SIZE);
    fprintf(stdout, "       GLOBAL_ELITE_POP_SIZE   : %d\n", GLOBAL_ELITE_POP_SIZE);
    fprintf(stdout, "[INFO] ========================================================\n");

    instance.input = &input;
    instance.current_scenario = &current_scenario;
    instance.etc = &etc;
    instance.energy = &energy;
    
    instance.population = (struct solution**)(malloc(sizeof(struct solution*) * input.thread_count));
    instance.local_elite_pop = (struct solution**)(malloc(sizeof(struct solution*) * input.thread_count));
    
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
            printf("Could not create slave thread %d\n", i);
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
    free(instance.population);
    free(instance.threads);
    free(threads);
}

/* Logica de los esclavos */
void* slave_thread(void *data) {
    struct cmochc_thread *t_data = (struct cmochc_thread*)data;
    struct cmochc *instance = t_data->data;
    
    // http://en.wikipedia.org/wiki/Merge_sort
    // http://en.wikipedia.org/wiki/Insertion_sort
    // http://blog.macuyiko.com/2009/01/modern-genetic-and-other-algorithms_237.html
    
    // ================================================================
    // Inicializo el thread.
    // ================================================================
    instance->population[t_data->thread_id] = (struct solution*)(malloc(sizeof(struct solution) * instance->input->population_size));
    
    for (int i = 0; i < instance->input->population_size; i++) {
        compute_mct_random(&(instance->population[t_data->thread_id][i]), i, i % 2);
        instance->population[t_data->thread_id][i].initialized = 1;
    }

    instance->local_elite_pop[t_data->thread_id] = (struct solution*)(malloc(sizeof(struct solution) * LOCAL_ELITE_POP_SIZE));

    for (int i = 0; i < LOCAL_ELITE_POP_SIZE; i++) {
        instance->local_elite_pop[t_data->thread_id][i].initialized = 0;
    }
    
    // ================================================================
    // .
    // ================================================================
    struct solution children
    
    for (int iteracion = 0; iteracion < LOCAL_ITERATION_COUNT; iteracion++) {
        
    }

    // ================================================================
    // Finalizo el thread.
    // ================================================================
    free(instance->population);
    free(instance->local_elite_pop);
    
    return 0;
}
