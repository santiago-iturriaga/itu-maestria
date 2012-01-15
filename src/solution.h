/*
 * Handles a solution in memory.
 */

#include <stdio.h>

#include "etc_matrix.h"

#ifndef SOLUTION_H_
#define SOLUTION_H_

#define TASK__NOT_ASSIGNED -1
#define       MACHINE__EOT -2
#define     MACHINE__EMPTY -1

struct solution {
    struct matrix *etc_matrix;
    
	int *__task_assignment;
    int **__machine_assignment;
    int *__machine_assignment_count;
    
	float *__machine_compute_time;	
	float __makespan;
};

struct solution* create_empty_solution(struct matrix *etc_matrix);
void clone_solution(struct solution *dst, struct solution *src);
void free_solution(struct solution *s);

void assign_task_to_machine(struct solution *s, int machine_id, int task_id);
void move_task_to_machine(struct solution *s, int machine_id, int task_id);
void swap_tasks(struct solution *s, int task_a_id, int task_b_id);

void refresh_makespan(struct solution *s);

int get_task_assignment(struct solution *s, int task_id);
float get_machine_compute_time(struct solution *s, int machine_id);
float get_makespan(struct solution *s);
int get_task_in_machine(struct solution *s, int position);
int* get_all_tasks(struct solution *s, int position);

void validate_solution(struct solution *s);
void show_solution(struct solution *s);

#endif
