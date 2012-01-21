/*
 * Handles a solution in memory.
 */

#include <stdio.h>

#include "etc_matrix.h"
#include "energy_matrix.h"

#ifndef SOLUTION_H_
#define SOLUTION_H_

#define SOLUTION__TASK_NOT_ASSIGNED -1

#define SOLUTION__STATUS_NOT_READY  -1
#define SOLUTION__STATUS_EMPTY      0
#define SOLUTION__STATUS_NEW        1
#define SOLUTION__STATUS_READY      2
#define SOLUTION__STATUS_TO_DEL     3

struct solution {
    struct etc_matrix *etc;
    struct energy_matrix *energy;
    
    int status;
    int initialized;
    
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

struct solution* create_empty_solution(struct etc_matrix *etc, struct energy_matrix *energy);
void init_empty_solution(struct etc_matrix *etc, struct energy_matrix *energy, struct solution *new_solution);

void clone_solution(struct solution *dst, struct solution *src, int clone_status);
void free_solution(struct solution *s);

void assign_task_to_machine(struct solution *s, int machine_id, int task_id);
void move_task_to_machine(struct solution *s, int task_id, int machine_id);
void move_task_to_machine_by_pos(struct solution *s, int machine_src, int task_src_pos, int machine_dst);
void swap_tasks(struct solution *s, int task_a_id, int task_b_id);
void swap_tasks_by_pos(struct solution *s, int machine_a, int task_a_pos, int machine_b, int task_b_pos);

void refresh_makespan(struct solution *s);
void refresh_energy(struct solution *s);
void refresh_worst_energy(struct solution *s);

int get_task_assigned_machine_id(struct solution *s, int task_id);
int get_machine_tasks_count(struct solution *s, int machine_id);
int get_machine_task_id(struct solution *s, int machine_id, int task_position);
int get_machine_task_pos(struct solution *s, int machine_id, int task_id);

float get_machine_compute_time(struct solution *s, int machine_id);
float get_makespan(struct solution *s);
float get_energy(struct solution *s);

int get_worst_ct_machine_id(struct solution *s);
int get_worst_energy_machine_id(struct solution *s);

void validate_solution(struct solution *s);
void show_solution(struct solution *s);

#endif
