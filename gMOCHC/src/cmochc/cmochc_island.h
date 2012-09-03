#if !defined(CMOCHC_ISLANDS__H)
#define CMOCHC_ISLANDS__H

#include "../config.h"
#include "../load_params.h"
#include "../scenario.h"
#include "../etc_matrix.h"
#include "../energy_matrix.h"

#define CMOCHC_LOCAL__MUTATE_INITIAL_POP

//#define CMOCHC_LOCAL__Z_FITNESS_NORM
#define CMOCHC_LOCAL__ZN_FITNESS_NORM

//#define CMOCHC_PARETO_FRONT__PATCHES        12
#define CMOCHC_PARETO_FRONT__PATCHES        24
//#define CMOCHC_PARETO_FRONT__PATCHES        48

//#define CMOCHC_PARETO_FRONT__FIXED_WEIGHTS
//#define CMOCHC_PARETO_FRONT__RANDOM_WEIGHTS
#define CMOCHC_PARETO_FRONT__ADAPT_AR_WEIGHTS
//#define CMOCHC_PARETO_FRONT__ADAPT_AM_WEIGHTS

#define CMOCHC_ARCHIVE__MAX_SIZE            50

#define CMOCHC_LOCAL__ITERATION_COUNT       50
#define CMOCHC_LOCAL__BEST_SOLS_KEPT        3
#define CMOCHC_LOCAL__MATING_MAX_THRESHOLD_DIVISOR  4
/* Aprox. one cataclysm every CROSS_THRESHOLD_STEP_DIVISOR local iterations without change */
#define CMOCHC_LOCAL__MATING_THRESHOLD_STEP_DIVISOR 15
#define CMOCHC_LOCAL__MATING_CHANCE         2
#define CMOCHC_LOCAL__MUTATE_CHANCE         4

#define CMOCHC_COLLABORATION__MOEAD_NEIGH_SIZE  4

#define CMOCHC_COLLABORATION__MIGRATION_BEST
//#define CMOCHC_COLLABORATION__MIGRATION_RANDOM_ELITE
//#define CMOCHC_COLLABORATION__MIGRATION_NONE

//#define CMOCHC_COLLABORATION__MIGRATE_BY_COPY
#define CMOCHC_COLLABORATION__MIGRATE_BY_MATE
//#define CMOCHC_COLLABORATION__MIGRATE_BY_MUTATE

void compute_cmochc_island(struct params &input, struct scenario &current_scenario, 
    struct etc_matrix &etc, struct energy_matrix &energy);

#endif // CMOCHC_ISLANDS__H
