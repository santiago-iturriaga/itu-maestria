#include <stdio.h>
#include <cuda.h>
#include <math.h>

#include "pals_serial.h"

void pals_serial(struct matrix *etc_matrix, struct solution *s, 
	int &best_swap_task_a, int &best_swap_task_b, float &best_swap_delta) {
	
	best_swap_task_a = -1;
	best_swap_task_b = -1;
	best_swap_delta = 0.0;

	for (int task_a = 0; task_a < etc_matrix->tasks_count; task_a++) {
		for (int task_b = 0; task_b < task_a; task_b++) {
			int machine_a = s->task_assignment[task_a];
			int machine_b = s->task_assignment[task_b];
			
			float current_swap_delta = 0.0;
			current_swap_delta -= get_etc_value(etc_matrix, machine_a, task_a);
			current_swap_delta -= get_etc_value(etc_matrix, machine_b, task_b);
			
			current_swap_delta += get_etc_value(etc_matrix, machine_a, task_b);
			current_swap_delta += get_etc_value(etc_matrix, machine_b, task_a);
			
			if (best_swap_delta > current_swap_delta) {
				best_swap_delta = current_swap_delta;
				best_swap_task_a = task_a;
				best_swap_task_b = task_b;
			}
		}
	}
}

