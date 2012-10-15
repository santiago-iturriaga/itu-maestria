#ifndef CMOCHC_ISLANDS__H
#define CMOCHC_ISLANDS__H

#include "../config.h"
#include "../load_params.h"
#include "../scenario.h"
#include "../etc_matrix.h"
#include "../energy_matrix.h"
#include "../config.h"
#include "../global.h"
#include "../solution.h"
#include "../utils.h"

#include "../basic/mct.h"
#include "../random/random.h"
#include "../archivers/aga.h"

/* ************************************************** */
/* Función que comienza la ejecución del algoritmo    */
/* ************************************************** */
void compute_cmochc_island();

/* ************************************************** */
/* Define el método de normalización de los objetivos */
/* ************************************************** */
//#define CMOCHC_LOCAL__Z_FITNESS_NORM
#define CMOCHC_LOCAL__ZN_FITNESS_NORM

/* ************************************************************************ */
/* Cuantos pesos diferentes se consideran para la discretización del frente */
/* ************************************************************************ */
//#define CMOCHC_PARETO_FRONT__PATCHES        8
//#define CMOCHC_PARETO_FRONT__PATCHES        12
//#define CMOCHC_PARETO_FRONT__PATCHES        24
//#define CMOCHC_PARETO_FRONT__PATCHES        32
#define CMOCHC_PARETO_FRONT__PATCHES        48
//#define CMOCHC_PARETO_FRONT__PATCHES        64

/* ******************************************************** */
/* Define la forma de asignación de los pesos a los workers */
/* ******************************************************** */
//#define CMOCHC_PARETO_FRONT__FIXED_WEIGHTS
//#define CMOCHC_PARETO_FRONT__RANDOM_WEIGHTS
//#define CMOCHC_PARETO_FRONT__ADAPT_AR_WEIGHTS
#define CMOCHC_PARETO_FRONT__ADAPT_AM_WEIGHTS

/* ************************************* */
/* Configuración del EA CHC de cada deme */
/* ************************************* */
/* Muta la población inicial creada con MCT */
#define CMOCHC_LOCAL__MUTATE_INITIAL_POP
#define CMOCHC_LOCAL__POPULATION_SIZE       12
#define CMOCHC_LOCAL__ITERATION_COUNT       750
#define CMOCHC_LOCAL__BEST_SOLS_KEPT        3
#define CMOCHC_LOCAL__MATING_MAX_THRESHOLD_DIVISOR  4
/* Aprox. one cataclysm every CROSS_THRESHOLD_STEP_DIVISOR local iterations without change */
#define CMOCHC_LOCAL__MATING_THRESHOLD_STEP_DIVISOR 15

/* ******************************************************************* */
/* Configuración de la migración de elementos desde el archivo al deme */
/* ******************************************************************* */
#define CMOCHC_COLLABORATION__MOEAD_NEIGH_SIZE  4

/* Define qué elementos son migrados */
#define CMOCHC_COLLABORATION__MIGRATION_BEST
//#define CMOCHC_COLLABORATION__MIGRATION_RANDOM_ELITE
//#define CMOCHC_COLLABORATION__MIGRATION_NONE

/* Define cómo son incluidos los elementos al deme */
#define CMOCHC_COLLABORATION__MIGRATE_BY_MATE
//#define CMOCHC_COLLABORATION__MIGRATE_BY_MUTATE
//#define CMOCHC_COLLABORATION__MIGRATE_BY_COPY

// Cantidad máxima de soluciones (padres+hijos)
#define MAX_POP_SOLS 2*CMOCHC_LOCAL__POPULATION_SIZE

#define CMOCHC_THREAD_STATUS__IDLE 0
#define CMOCHC_THREAD_STATUS__CHC_FROM_NEW 1
#define CMOCHC_THREAD_STATUS__CHC_FROM_ARCHIVE 2
#define CMOCHC_THREAD_STATUS__LS 3
#define CMOCHC_THREAD_STATUS__STOP 4

#define CMOCHC_MASTER_STATUS__CHC 0
#define CMOCHC_MASTER_STATUS__LS 1

struct cmochc_thread {
    /* Id del esclavo */
    int thread_id;
       
    /* Poblacion de cada esclavo */
    struct solution population[MAX_POP_SOLS];
    int sorted_population[MAX_POP_SOLS];
    FLOAT fitness_population[MAX_POP_SOLS];

    /* Merge sort tmp array */
    int merge_sort_tmp[MAX_POP_SOLS];

    FLOAT weight_makespan;
    FLOAT weight_energy;

    FLOAT makespan_zenith_value, energy_zenith_value;
    FLOAT makespan_nadir_value, energy_nadir_value;
    
    int migration_global_pop_index[CMOCHC_COLLABORATION__MOEAD_NEIGH_SIZE];
    int migration_current_weight_distance[CMOCHC_COLLABORATION__MOEAD_NEIGH_SIZE];
    
    int threshold_max;
    int threshold_step;
};

extern struct cmochc_thread EA_THREADS[MAX_THREADS];

struct cmochc_island {
    /* Coleccion de esclavos */
    pthread_t threads[MAX_THREADS];

    /* Estado de cada hilo */
    int thread_status[MAX_THREADS];
    int thread_idle_count;
    
    /* Descomposición del frente de pareto */
    FLOAT weights[CMOCHC_PARETO_FRONT__PATCHES];
    int thread_weight_assignment[MAX_THREADS];
    int weight_thread_assignment[CMOCHC_PARETO_FRONT__PATCHES];
    
    /* Random generator de cada esclavo y para el master */
    RAND_STATE rand_state[MAX_THREADS+1];
    
    /* Archiver */
    int archiver_new_pop_size;

    /* Sync */
    pthread_barrier_t sync_barrier;
    pthread_mutex_t status_cond_mutex;
    pthread_cond_t worker_status_cond;
    pthread_cond_t master_status_cond;

    /* Aux master thread memory */
    int weight_gap_count;
    int weight_gap_sorted[ARCHIVER__MAX_SIZE + MAX_THREADS + 1];
    int weight_gap_length[ARCHIVER__MAX_SIZE + MAX_THREADS + 1];
    int weight_gap_index[ARCHIVER__MAX_SIZE + MAX_THREADS + 1];
    int weight_gap_tmp[ARCHIVER__MAX_SIZE + MAX_THREADS + 1];
};

extern struct cmochc_island EA_INSTANCE;

/* Statistics */
#ifdef DEBUG_1
    extern int COUNT_GENERATIONS[MAX_THREADS];
    
    /* Al menos una nueva solución fue agregada a la población de 
     * padres del CHC del deme luego de aplicar una iteración de crossovers */
    extern int COUNT_AT_LEAST_ONE_CHILDREN_INSERTED[MAX_THREADS]; 
    /* Cantidad de veces que el CHC del deme mejoró la mejor solución que tenía */
    extern int COUNT_IMPROVED_BEST_SOL[MAX_THREADS]; 
    /* Cantidad de crossovers aplicados */
    extern int COUNT_CROSSOVER[MAX_THREADS]; 
    /* Cantidad de crossovers que produjeron al menos uno de los hijos 
     * mejor a alguno de sus padres */
    extern int COUNT_IMPROVED_CROSSOVER[MAX_THREADS]; 
    /* Cantidad de soluciones mutadas en cataclismos */
    extern int COUNT_CATACLYSM[MAX_THREADS]; 
    /* Cantidad de soluciones mejoradas durante cataclismos */
    extern int COUNT_IMPOVED_CATACLYSM[MAX_THREADS]; 
    /* Cantidad de ejecuciones del método de migracion */
    extern int COUNT_MIGRATIONS[MAX_THREADS]; 
    /* Cantidad de soluciones migradas */
    extern int COUNT_SOLUTIONS_MIGRATED[MAX_THREADS]; 
    /* Cantidad histórica de ocurrencias de cada peso en el archivo AGA */
    extern int COUNT_HISTORIC_WEIGHTS[CMOCHC_PARETO_FRONT__PATCHES]; 
#endif

#endif // CMOCHC_ISLANDS__H
