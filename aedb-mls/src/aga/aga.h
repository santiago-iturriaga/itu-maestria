#include "../config.h"
#include "../solution.h"
#include "../random/cpu_mt.h"

#ifndef AGA__H_
#define AGA__H_

/* This is the number of recursive subdivisions of the objective space carried out in order to
 * divide the objective space into a grid for the purposes of diversity maintenance. Values of
 * between 3 and 6 are useful, depending on number of objectives.
 * */
#define ARCHIVER__AGA_DEPTH 6

/* About the maximum size of an integer for your compiler. */
#define ARCHIVER__AGA_LARGE 2000000000

#define ARCHIVER__MAX_INPUT_BUFFER MLS__MAX_THREADS

struct aga_state {
    int max_locations; // Number of locations in grid.
    
    double gl_offset[OBJECTIVES]; // The range, offset etc of the grid.
    double gl_range[OBJECTIVES];
    double gl_largest[OBJECTIVES];
    
    int *grid_pop;
    int grid_sol_loc[AGA__MAX_ARCHIVE_SIZE];
    
    struct solution population[AGA__MAX_ARCHIVE_SIZE];
    int population_count;
    
    struct cpu_mt_state random_state;
    struct solution aux_population[MLS__MAX_THREADS];
};

extern struct aga_state AGA;

/*
 * Ejecuta el algoritmo.
 */
void archivers_aga(int seed);

#ifdef NONMPI
    int archivers_aga_add(int new_solution_pos);
#endif
    
#endif // AGA__H_
