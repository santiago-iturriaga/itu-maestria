#include "../config.h"
#include "../utils.h"

#include "../config.h"
#include "minmin.h"

void compute_minmin(struct solution *solution) {
	if (DEBUG) fprintf(stdout, "[DEBUG] calculando MinMin...\n");

	// Timming -----------------------------------------------------
	timespec ts;
	timming_start(ts);
	// Timming -----------------------------------------------------

	int assigned_tasks[solution->etc->tasks_count];
	for (int i = 0; i < solution->etc->tasks_count; i++) {
		assigned_tasks[i] = 0;
	}

	int assigned_tasks_count = 0;
	
	while (assigned_tasks_count < solution->etc->tasks_count) { // Mientras quede una tarea sin asignar.
		int best_task;
		int best_machine;
		float best_cost;
		
		best_task = -1;
		best_machine = -1;
		best_cost = 0.0;
	
		// Recorro las tareas.
		for (int task_i = 0; task_i < solution->etc->tasks_count; task_i++) {			
			// Si la tarea task_i no esta asignada.
			if (assigned_tasks[task_i] == 0) {
				int best_machine_for_task;
				best_machine_for_task = 0;
				
				float best_machine_cost_for_task;
				
				best_machine_cost_for_task = get_machine_compute_time(solution, 0) + 
					get_etc_value(solution->etc, 0, task_i);
			
				for (int machine_x = 1; machine_x < solution->etc->machines_count; machine_x++) {
					float current_cost;
					
					current_cost = get_machine_compute_time(solution, machine_x) + 
						get_etc_value(solution->etc, machine_x, task_i);
				
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
		
		assigned_tasks_count++;
		assigned_tasks[best_task] = 1;
		
		assign_task_to_machine(solution, best_machine, best_task);
	
		/*if (DEBUG) {
			fprintf(stdout, "[DEBUG] best_machine: %d, best_task: %d.\n", best_machine, best_task);
		}*/
		
	    refresh_makespan(solution);
        refresh_energy(solution);
	}
	
	if (DEBUG) fprintf(stdout, "[DEBUG] MinMin Solution >> makespan: %f || energy: %f.\n", get_makespan(solution), get_energy(solution));
	//else fprintf(stdout, "%f|%f\n", get_makespan(solution), get_energy(solution));

	// Timming -----------------------------------------------------
	timming_end("MinMin time", ts);
	// Timming -----------------------------------------------------
}
