/*
 * Handles a solution in memory.
 */

#include <stdio.h>

#include "etc_matrix.h"

#ifndef SOLUTION_H_
#define SOLUTION_H_

struct solution {
	ushort *task_assignment;
	float *machine_compute_time;	
	float makespan;
};

struct solution* create_empty_solution(struct matrix *etc_matrix);
void clone_solution(struct matrix *etc_matrix, struct solution *dst, struct solution *src);
void free_solution(struct solution *s);

void validate_solution(struct matrix *etc_matrix, struct solution *s);
void show_solution(struct matrix *etc_matrix, struct solution *s);

#endif
