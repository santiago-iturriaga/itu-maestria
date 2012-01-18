/*
 * Handles a solution in memory.
 */

#include <stdio.h>

#include "etc_matrix.h"

#ifndef SOLUTION_H_
#define SOLUTION_H_

#define TASK__NOT_ASSIGNED -1

struct solution {
    struct etc_matrix *etc;
    
	int *__task_assignment;
    int **__machine_assignment;
    int *__machine_assignment_count;
    
    // Makespan
	float *__machine_compute_time;	
	int __worst_ct_machine_id;
	float __makespan;
	
	// Energy
	float *__machine_energy_consumption;
	int __worst_energy_machine_id;
	float __total_energy_consumption;
};

struct solution* create_empty_solution(struct etc_matrix *etc);
void init_empty_solution(struct etc_matrix *etc, struct solution *new_solution);

void clone_solution(struct solution *dst, struct solution *src);
void free_solution(struct solution *s);

void assign_task_to_machine(struct solution *s, int machine_id, int task_id);
void move_task_to_machine(struct solution *s, int task_id, int machine_id);
void swap_tasks(struct solution *s, int task_a_id, int task_b_id);

void refresh_makespan(struct solution *s);

int get_task_assigned_machine_id(struct solution *s, int task_id);
int get_machine_tasks_count(struct solution *s, int machine_id);
int get_machine_task_id(struct solution *s, int machine_id, int task_position);
int get_machine_task_pos(struct solution *s, int machine_id, int task_id);

float get_machine_compute_time(struct solution *s, int machine_id);
float get_makespan(struct solution *s);

void validate_solution(struct solution *s);
void show_solution(struct solution *s);

#endif
