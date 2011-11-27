#include <assert.h>

#include "solution.h"

struct solution* create_empty_solution(struct matrix *etc_matrix) {
	struct solution *new_solution;
	new_solution = (struct solution*)(malloc(sizeof(struct solution)));
	
	if (new_solution == NULL) {
		fprintf(stderr, "[ERROR] Solicitando memoria para new_solution.\n");
		exit(EXIT_FAILURE);
	}
	
	new_solution->makespan = 0.0;
	new_solution->task_assignment = (int*)(malloc(sizeof(int) * etc_matrix->tasks_count));
	
	if (new_solution->task_assignment == NULL) {
		fprintf(stderr, "[ERROR] Solicitando memoria para el new_solution->task_assignment.\n");
		exit(EXIT_FAILURE);
	}
	
	for (int task = 0; task < etc_matrix->tasks_count; task++) {
		new_solution->task_assignment[task] = -1; /* not yet assigned */
	}

	new_solution->machine_compute_time = (float*)(malloc(sizeof(float) * etc_matrix->machines_count));
	
	if (new_solution->machine_compute_time == NULL) {
		fprintf(stderr, "[ERROR] Solicitando memoria para el new_solution->machine_compute_time.\n");
		exit(EXIT_FAILURE);
	}
	
	for (int machine = 0; machine < etc_matrix->machines_count; machine++) {
		new_solution->machine_compute_time[machine] = 0.0;
	}
	
	return new_solution;
}

void free_solution(struct solution *s) {
	free(s->task_assignment);
	free(s->machine_compute_time);
	free(s);
}

void validate_solution(struct matrix *etc_matrix, struct solution *s) {
	fprintf(stdout, "[INFO] Validate solution =========================== \n");
	fprintf(stdout, "[INFO] Dimension (%d x %d).\n", etc_matrix->tasks_count, etc_matrix->machines_count);
	fprintf(stdout, "[INFO] Makespan: %f.\n", s->makespan);
	
	{
		float aux_makespan = 0.0;

		for (int machine = 0; machine < etc_matrix->machines_count; machine++) {
			if (aux_makespan < s->machine_compute_time[machine]) {
				aux_makespan = s->machine_compute_time[machine];
			}
		}
	
		assert(s->makespan == aux_makespan);
	}
	
	for (int machine = 0; machine < etc_matrix->machines_count; machine++) {
		float aux_compute_time;
		aux_compute_time = 0.0;
	
		for (int task = 0; task < etc_matrix->tasks_count; task++) {
			if (s->task_assignment[task] == machine) {
				aux_compute_time += get_etc_value(etc_matrix, machine, task);
			}
		}
		
		assert(s->machine_compute_time[machine] == aux_compute_time);
	}
	
	for (int task = 0; task < etc_matrix->tasks_count; task++) {
		assert(s->task_assignment[task] >= 0);
		assert(s->task_assignment[task] < etc_matrix->machines_count);
	}
	
	fprintf(stdout, "[INFO] The current solution is valid.\n");
	fprintf(stdout, "[INFO] ============================================= \n");	
}

void show_solution(struct matrix *etc_matrix, struct solution *s) {
	fprintf(stdout, "[INFO] Show solution =========================== \n");
	fprintf(stdout, "[INFO] Dimension (%d x %d).\n", etc_matrix->tasks_count, etc_matrix->machines_count);

	fprintf(stdout, "   Makespan: %f.\n", s->makespan);
	
	for (int machine = 0; machine < etc_matrix->machines_count; machine++) {
		fprintf(stdout, "   Machine: %d -> execution time: %f.\n", machine, s->machine_compute_time[machine]);
	}
	
	for (int task = 0; task < etc_matrix->tasks_count; task++) {
		fprintf(stdout, "   Task: %d -> assigned to: %d.\n", task, s->task_assignment[task]);
	}
	fprintf(stdout, "[INFO] ========================================= \n");
}
