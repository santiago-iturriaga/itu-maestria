#include "../config.h"
#include "mct.h"

void compute_mct(struct matrix *etc_matrix, struct solution *solution) {
	for (int task = 0; task < etc_matrix->tasks_count; task++) {
		int best_machine;
		best_machine = 0;
		
		float best_etc_value;
		best_etc_value = get_etc_value(etc_matrix, 0, task);
	
		for (int machine = 1; machine < etc_matrix->machines_count; machine++) {
			float etc_value;
			etc_value = get_etc_value(etc_matrix, machine, task);
			
			if (get_machine_compute_time(solution, machine) + etc_value < 
				get_machine_compute_time(solution, best_machine) + best_etc_value) {
				
				best_etc_value = etc_value;
				best_machine = machine;
			}
		}
		
		assign_task_to_machine(solution, best_machine, task);
	}
	
	if (DEBUG) fprintf(stdout, "[DEBUG] MCT Solution makespan: %f.\n", get_makespan(solution));
}
