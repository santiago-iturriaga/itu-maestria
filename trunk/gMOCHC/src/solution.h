/*
 * Handles a solution in memory.
 */

#include <stdio.h>

#include "scenario.h"
#include "etc_matrix.h"
#include "energy_matrix.h"

#ifndef SOLUTION_H_
#define SOLUTION_H_

#define SOLUTION__TASK_NOT_ASSIGNED -1

struct solution {
    int initialized;
    
    struct scenario *s;
    struct etc_matrix *etc;
    struct energy_matrix *energy;
    
    int *task_assignment;
    
    float *machine_compute_time;
    float makespan;
    float *machine_energy_consumption;
    float energy_consumption;
};

void create_empty_solution(struct solution *new_solution, struct scenario *s, struct etc_matrix *etc, struct energy_matrix *energy);
void clone_solution(struct solution *dst, struct solution *src);
void free_solution(struct solution *s);

void refresh_solution(struct solution *s);
void validate_solution(struct solution *s);
void show_solution(struct solution *s);

#endif
