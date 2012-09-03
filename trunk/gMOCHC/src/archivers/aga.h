#include "../config.h"
#include "../solution.h"

#ifndef AGA__H_
#define AGA__H_

#define ARCHIVER__OBJECTIVES 2

/* This is the number of recursive subdivisions of the objective space carried out in order to
 * divide the objective space into a grid for the purposes of diversity maintenance. Values of
 * between 3 and 6 are useful, depending on number of objectives.
 * */
#define ARCHIVER__AGA_DEPTH 6

/* About the maximum size of an integer for your compiler. */
#define ARCHIVER__AGA_LARGE 2000000000

struct aga_state {
    int max_locations; // Number of locations in grid.
    
    FLOAT gl_offset[ARCHIVER__OBJECTIVES]; // The range, offset etc of the grid.
    FLOAT gl_range[ARCHIVER__OBJECTIVES];
    FLOAT gl_largest[ARCHIVER__OBJECTIVES];
    
    int *grid_pop;
    int *grid_sol_loc;
    
    struct solution *population;
    int *population_tag;
    int population_size;
    int population_count;
    
    int tag_max;
    int *tag_count;
    
    struct solution *new_solutions;
    int *new_solutions_tag;
    int new_solutions_size;
};

void archivers_aga_init(struct aga_state *state, int population_max_size,
    struct solution *new_solutions, int *new_solutions_tag, int new_solutions_size,
    int tag_max);   
    
void archivers_aga_free(struct aga_state *state);

int archivers_aga(struct aga_state *state);

void archivers_aga_show(struct aga_state *state);
void archivers_aga_dump(struct aga_state *state);

#endif // AGA__H_
