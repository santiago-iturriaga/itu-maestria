#if !defined(CMOCHC_ISLANDS__H)
#define CMOCHC_ISLANDS__H

#include "../load_params.h"
#include "../scenario.h"
#include "../etc_matrix.h"
#include "../energy_matrix.h"

/* For debugging purposes */
//#define CMOCHC_SYNC

#define CMOCHC_LOCAL__ITERATION_COUNT 50
#define CMOCHC_LOCAL__BEST_SOLS_KEPT 3
#define CMOCHC_LOCAL__MATING_MAX_THRESHOLD_DIVISOR 4
/* Aprox. one cataclysm every CROSS_THRESHOLD_STEP_DIVISOR local iterations without change */
#define CMOCHC_LOCAL__MATING_THRESHOLD_STEP_DIVISOR 15
#define CMOCHC_LOCAL__MATING_CHANCE 2
#define CMOCHC_LOCAL__MUTATE_CHANCE 3

#define CMOCHC_ARCHIVE__MAX_SIZE 50

#define CMOCHC_COLLABORATION__MOEAD_NEIGH_SIZE 4

//#define CMOCHC_COLLABORATION__MIGRATION_BEST
#define CMOCHC_COLLABORATION__MIGRATION_RANDOM_ELITE
//#define CMOCHC_COLLABORATION__MIGRATION_NONE

//#define CMOCHC_COLLABORATION__MIGRATE_BY_COPY
#define CMOCHC_COLLABORATION__MIGRATE_BY_MATE
//#define CMOCHC_COLLABORATION__MIGRATE_BY_MUTATE

#define CMOCHC_COLLABORATION__MUTATE_BEST
//#define CMOCHC_COLLABORATION__MUTATE_ALL_ELITE
//#define CMOCHC_COLLABORATION__MUTATE_NONE

void compute_cmochc_island(struct params &input, struct scenario &current_scenario, 
    struct etc_matrix &etc, struct energy_matrix &energy);

#endif // CMOCHC_ISLANDS__H
