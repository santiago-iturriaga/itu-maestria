#include "../solution.h"
#include "../mls/mls.h"

#ifndef AGA__H_
#define AGA__H_

/* This is the number of recursive subdivisions of the objective space carried out in order to
 * divide the objective space into a grid for the purposes of diversity maintenance. Values of
 * between 3 and 6 are useful, depending on number of objectives.
 * */
#define ARCHIVER__AGA_DEPTH 6

/* About the maximum size of an integer for your compiler. */
#define ARCHIVER__AGA_LARGE 2000000000

struct aga_state {
    int max_locations; // Number of locations in grid.
    
    double gl_offset[OBJECTIVES]; // The range, offset etc of the grid.
    double gl_range[OBJECTIVES];
    double gl_largest[OBJECTIVES];
    
    int *grid_pop;
    int *grid_sol_loc;
};

void archivers_aga_init(struct mls_instance *instance);
void archivers_aga_free(struct mls_instance *instance);
int archivers_aga(struct mls_thread_arg *instance, int new_solution_pos);

#endif // AGA__H_
