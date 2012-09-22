#include "../config.h"
#include "../solution.h"

#ifndef AGA__H_
#define AGA__H_

//#define ARCHIVER__MAX_SIZE 25
#define ARCHIVER__MAX_SIZE 50

#define ARCHIVER__MAX_NEW_SOLS MAX_THREADS*4
#define ARCHIVER__MAX_TAGS 512
#define ARCHIVER__OBJECTIVES 2

/* This is the number of recursive subdivisions of the objective space carried out in order to
 * divide the objective space into a grid for the purposes of diversity maintenance. Values of
 * between 3 and 6 are useful, depending on number of objectives.
 * */
#define ARCHIVER__AGA_DEPTH 3
//#define ARCHIVER__AGA_DEPTH 6

/* About the maximum size of an integer for your compiler. */
#define ARCHIVER__AGA_LARGE 2000000000

struct aga_state {
    int max_locations; // Number of locations in grid.
    
    FLOAT gl_offset[ARCHIVER__OBJECTIVES]; // The range, offset etc of the grid.
    FLOAT gl_range[ARCHIVER__OBJECTIVES];
    FLOAT gl_largest[ARCHIVER__OBJECTIVES];
    
    int *grid_pop;
    int *grid_sol_loc;
    
    struct solution population[ARCHIVER__MAX_SIZE];
    int population_tag[ARCHIVER__MAX_SIZE];
    int population_count;
    
    int tag_count[ARCHIVER__MAX_TAGS];
    int tag_size;
    
    struct solution new_solutions[ARCHIVER__MAX_NEW_SOLS];
    int new_solutions_tag[ARCHIVER__MAX_NEW_SOLS];
};

extern struct aga_state ARCHIVER;

void archivers_aga_init(int tag_size);
void archivers_aga_free(int new_solutions_size);

int archivers_aga(int new_solutions_size);

void archivers_aga_show();
void archivers_aga_dump();

#endif // AGA__H_
