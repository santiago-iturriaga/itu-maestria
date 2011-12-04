#include "mct.h"

void compute_mct(struct matrix *etc_matrix, struct solution *solution) {
	solution->makespan = 0.0;

	for (int task = 0; task < etc_matrix->tasks_count; task++) {
		int best_machine;
		best_machine = 0;
		
		float best_etc_value;
		best_etc_value = get_etc_value(etc_matrix, 0, task);
	
		for (int machine = 1; machine < etc_matrix->machines_count; machine++) {
			float etc_value;
			etc_value = get_etc_value(etc_matrix, machine, task);
			
			if (solution->machine_compute_time[machine] + etc_value < 
				solution->machine_compute_time[best_machine] + best_etc_value) {
				
				best_etc_value = etc_value;
				best_machine = machine;
			}
		}
		
		solution->task_assignment[task] = best_machine;
		solution->machine_compute_time[best_machine] += best_etc_value;
		
		if (solution->machine_compute_time[best_machine] > solution->makespan) {
			solution->makespan = solution->machine_compute_time[best_machine];
		}
	}
}
