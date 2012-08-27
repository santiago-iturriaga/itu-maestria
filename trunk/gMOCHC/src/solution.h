/*
 * Handles a solution in memory.
 */

#include <stdio.h>

#include "config.h"
#include "scenario.h"
#include "etc_matrix.h"
#include "energy_matrix.h"

#ifndef SOLUTION_H_
#define SOLUTION_H_

#define SOLUTION__NOT_INITIALIZED -1
#define SOLUTION__EMPTY 0
#define SOLUTION__IN_USE 1
#define SOLUTION__TASK_NOT_ASSIGNED -1

struct solution {
    int initialized;
    
    struct scenario *s;
    struct etc_matrix *etc;
    struct energy_matrix *energy;
    
    int *task_assignment;
    
    FLOAT *machine_compute_time;
    FLOAT makespan;
    
    FLOAT *machine_energy_consumption;
    FLOAT energy_consumption;
};

void create_empty_solution(struct solution *new_solution, struct scenario *s, struct etc_matrix *etc, struct energy_matrix *energy);
void clone_solution(struct solution *dst, struct solution *src);
void free_solution(struct solution *s);

void refresh_solution(struct solution *s);
void validate_solution(struct solution *s);
void show_solution(struct solution *s);

#endif
