#include "mct.h"

void compute_minmin(struct matrix *etc_matrix, struct solution *solution) {
	short assigned_tasks[etc_matrix->tasks_count];
	for (int i = 0; i < etc_matrix->tasks_count; i++) {
		assigned_tasks[i] = 0;
	}

	int assigned_tasks_count = 0;
	
	while (assigned_tasks_count < etc_matrix->tasks_count) { // Mientras quede una tarea sin asignar.
		int best_task;
		int best_machine;
		float best_cost;
		
		best_task = -1;
		best_machine = -1;
		best_cost = 0.0;
	
		// Recorro las tareas.
		for (int task_i = 0; task_i < etc_matrix->tasks_count; task_i++) {			
			// Si la tarea task_i no esta asignada.
			if (assigned_tasks[task_i] == 0) {
				int best_machine_for_task;
				best_machine_for_task = 0;
				
				float best_machine_cost_for_task;
				
				best_machine_cost_for_task = solution->machine_compute_time[0] + 
					get_etc_value(etc_matrix, 0, task_i);
			
				for (int machine_x = 1; machine_x < etc_matrix->machines_count; machine_x++) {
					float current_cost;
					
					current_cost = solution->machine_compute_time[machine_x] + 
						get_etc_value(etc_matrix, machine_x, task_i);
				
					if (current_cost < best_machine_cost_for_task) {
						best_machine_cost_for_task = current_cost;
						best_machine_for_task = machine_x;
					}
				}
			
				if ((best_machine_cost_for_task < best_cost) || (best_task < 0)) {
					best_task = task_i;
					best_machine = best_machine_for_task;
					best_cost = best_machine_cost_for_task;
				}
			}
		}
		
		assigned_tasks_count--;
		assigned_tasks[best_task] = 1;
		
		solution->task_assignment[best_task] = best_machine;
		
		solution->machine_compute_time[best_machine] = solution->machine_compute_time[best_machine] + 
			get_etc_value(etc_matrix, best_machine, best_task);
	}
}
