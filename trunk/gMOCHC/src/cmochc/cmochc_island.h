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
#define CMOCHC_LOCAL__ITERATION_COUNT       50
#define CMOCHC_LOCAL__BEST_SOLS_KEPT        3
#define CMOCHC_LOCAL__MATING_MAX_THRESHOLD_DIVISOR  4
/* Aprox. one cataclysm every CROSS_THRESHOLD_STEP_DIVISOR local iterations without change */
#define CMOCHC_LOCAL__MATING_THRESHOLD_STEP_DIVISOR 15
#define CMOCHC_LOCAL__MATING_CHANCE         2
#define CMOCHC_LOCAL__MUTATE_CHANCE         4

/* ******************************************************************* */
/* Configuración de la migración de elementos desde el archivo al deme */
/* ******************************************************************* */
#define CMOCHC_COLLABORATION__MOEAD_NEIGH_SIZE  4
/* Define qué elementos son migrados */
#define CMOCHC_COLLABORATION__MIGRATION_BEST
//#define CMOCHC_COLLABORATION__MIGRATION_RANDOM_ELITE
//#define CMOCHC_COLLABORATION__MIGRATION_NONE
//#define CMOCHC_COLLABORATION__MIGRATE_BY_COPY
/* Define cómo son incluidos los ementos al deme */
#define CMOCHC_COLLABORATION__MIGRATE_BY_MATE
//#define CMOCHC_COLLABORATION__MIGRATE_BY_MUTATE

void compute_cmochc_island();

// Cantidad máxima de soluciones (padres+hijos)
#define MAX_POP_SOLS 2*CMOCHC_LOCAL__POPULATION_SIZE

struct cmochc_thread {
    /* Id del esclavo */
    int thread_id;
    
    /* Poblacion de cada esclavo */
    struct solution population[MAX_POP_SOLS];
    int sorted_population[MAX_POP_SOLS];
    FLOAT fitness_population[MAX_POP_SOLS];

    /* Merge sort tmp array */
    int merge_sort_tmp[MAX_POP_SOLS];

    int currently_assigned_weight;
    FLOAT weight_makespan;
    FLOAT energy_makespan;

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
    
    /* Descomposición del frente de pareto */
    FLOAT weights[CMOCHC_PARETO_FRONT__PATCHES];
    int thread_weight_assignment[MAX_THREADS];
    int weight_thread_assignment[CMOCHC_PARETO_FRONT__PATCHES];
    
    int stopping_condition;

    /* Random generator de cada esclavo y para el master */
    RAND_STATE rand_state[MAX_THREADS+1];

    /* Sync */
    pthread_barrier_t sync_barrier;

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
