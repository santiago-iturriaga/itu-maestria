#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "config.h"
#include "solution.h"

struct solution* create_empty_solution(struct matrix *etc_matrix) {
	struct solution *new_solution;
	new_solution = (struct solution*)(malloc(sizeof(struct solution)));
	
	if (new_solution == NULL) {
		fprintf(stderr, "[ERROR] Solicitando memoria para new_solution.\n");
		exit(EXIT_FAILURE);
	}

    new_solution->etc_matrix = etc_matrix;
	new_solution->__makespan = 0.0;
	
	//=== Estructura orientada a tareas.
	new_solution->__task_assignment = (int*)(malloc(sizeof(int) * etc_matrix->tasks_count));
	
	if (new_solution->__task_assignment == NULL) {
		fprintf(stderr, "[ERROR] Solicitando memoria para el new_solution->task_assignment.\n");
		exit(EXIT_FAILURE);
	}
	
	for (int task = 0; task < etc_matrix->tasks_count; task++) {
		new_solution->__task_assignment[task] = TASK__NOT_ASSIGNED; /* not yet assigned */
	}
	
	//=== Estructura orientada a máquinas.
	new_solution->__machine_assignment = (int**)(malloc(sizeof(int*) * etc_matrix->machines_count));
	
	if (new_solution->__machine_assignment == NULL) {
		fprintf(stderr, "[ERROR] Solicitando memoria para el new_solution->machine_assignment.\n");
		exit(EXIT_FAILURE);
	}
	
	for (int machine = 0; machine < etc_matrix->machines_count; machine++) {
		new_solution->__machine_assignment[machine] = (int*)(malloc(sizeof(int) * etc_matrix->tasks_count));
		
		if (new_solution->__machine_assignment[machine] == NULL) {
		    fprintf(stderr, "[ERROR] Solicitando memoria para el new_solution->__machine_assignment[%d].\n", machine);
		    exit(EXIT_FAILURE);
	    }
	
	    new_solution->__machine_assignment[machine][0] = MACHINE__EOT;
	}	
	
	new_solution->__machine_assignment_count = (int*)(malloc(sizeof(int) * etc_matrix->machines_count));
	
	for (int machine = 0; machine < etc_matrix->machines_count; machine++) {
	    new_solution->__machine_assignment_count[machine] = 0;
	}
	
	//=== Estructura de machine compute time.
	new_solution->__machine_compute_time = (float*)(malloc(sizeof(float) * etc_matrix->machines_count));
	
	if (new_solution->__machine_compute_time == NULL) {
		fprintf(stderr, "[ERROR] Solicitando memoria para el new_solution->__machine_compute_time.\n");
		exit(EXIT_FAILURE);
	}
	
	for (int machine = 0; machine < etc_matrix->machines_count; machine++) {
		new_solution->__machine_compute_time[machine] = 0.0;
	}
	
	return new_solution;
}

void clone_solution(struct solution *dst, struct solution *src) {
	dst->__makespan = src->__makespan;
	memcpy(dst->__task_assignment, src->__task_assignment, sizeof(int) * src->etc_matrix->tasks_count);
	memcpy(dst->__machine_compute_time, src->__machine_compute_time, sizeof(float) * src->etc_matrix->machines_count);
}

void free_solution(struct solution *s) {
	free(s->__task_assignment);
	free(s->__machine_assignment_count);
	for (int machine = 0; machine < s->etc_matrix->machines_count; machine++) {
		free(s->__machine_assignment[machine]);
	}
	free(s->__machine_assignment);
	free(s->__machine_compute_time);
	free(s);
}

void assign_task_to_machine(struct solution *s, int machine_id, int task_id) {
    assert(machine_id < s->etc_matrix->machines_count);
    assert(machine_id >= 0);
    assert(task_id < s->etc_matrix->tasks_count);
    assert(task_id >= 0);
    assert(s->__task_assignment[task_id] == TASK__NOT_ASSIGNED);
       
    s->__machine_compute_time[machine_id] += get_etc_value(s->etc_matrix, machine_id, task_id);
    if (s->__machine_compute_time[machine_id] > s->__makespan) s->__makespan = s->__machine_compute_time[machine_id];
    
    s->__task_assignment[task_id] = machine_id;
    
    if (s->__machine_assignment_count[machine_id] > 0) {
        if (s->__machine_assignment[machine_id][s->__machine_assignment_count[machine_id]] == MACHINE__EOT) {
            s->__machine_assignment[machine_id][s->__machine_assignment_count[machine_id]] = task_id;
            s->__machine_assignment[machine_id][s->__machine_assignment_count[machine_id] + 1] = MACHINE__EOT;
        } else {
            int ready = 0;
            for (int i = 0; (i < s->etc_matrix->tasks_count) 
                && (s->__machine_assignment[machine_id][i] != MACHINE__EOT)
                && (ready == 0); i++) {
                
                if (s->__machine_assignment[machine_id][i] == MACHINE__EMPTY) {
                    s->__machine_assignment[machine_id][i] = task_id;                
                    ready = 1;
                }
            }
        }
    } else {
        s->__machine_assignment[machine_id][0] = task_id;
        s->__machine_assignment[machine_id][1] = MACHINE__EOT;    
    }
    
    s->__machine_assignment_count[machine_id] = s->__machine_assignment_count[machine_id] + 1;
}

void move_task_to_machine(struct solution *s, int machine_id, int task_id) {
    assert(machine_id < s->etc_matrix->machines_count);
    assert(machine_id >= 0);
    assert(task_id < s->etc_matrix->tasks_count);
    assert(task_id >= 0);
    assert(s->__task_assignment[task_id] != TASK__NOT_ASSIGNED);
    assert(s->__task_assignment[task_id] != machine_id);
    
    int recompute_makespan = 0;    
    int machine_origin_id = s->__task_assignment[task_id];
    
    s->__machine_compute_time[machine_id] += get_etc_value(s->etc_matrix, machine_id, task_id);
    
    if (s->__machine_compute_time[machine_id] > s->__makespan) {
        s->__makespan = s->__machine_compute_time[machine_id];
    } else if (s->__machine_compute_time[machine_origin_id] == s->__makespan) {
        recompute_makespan = 1;
    }
    
    s->__machine_compute_time[machine_origin_id] -= get_etc_value(s->etc_matrix, machine_origin_id, task_id);
    
    // Agrego la task a la máquina destino.
    s->__task_assignment[task_id] = machine_id;
    
    if (s->__machine_assignment_count[machine_id] > 0) {
        if (s->__machine_assignment[machine_id][s->__machine_assignment_count[machine_id]] == MACHINE__EOT) {
            s->__machine_assignment[machine_id][s->__machine_assignment_count[machine_id]] = task_id;
            s->__machine_assignment[machine_id][s->__machine_assignment_count[machine_id] + 1] = MACHINE__EOT;
        } else {
            int ready = 0;
            for (int i = 0; (i < s->etc_matrix->tasks_count) 
                && (s->__machine_assignment[machine_id][i] != MACHINE__EOT)
                && (ready == 0); i++) {
                
                if (s->__machine_assignment[machine_id][i] == MACHINE__EMPTY) {
                    s->__machine_assignment[machine_id][i] = task_id;                
                    ready = 1;
                }
            }
        }
    } else {
        s->__machine_assignment[machine_id][0] = task_id;
        s->__machine_assignment[machine_id][1] = MACHINE__EOT;    
    }
    
    s->__machine_assignment_count[machine_id] = s->__machine_assignment_count[machine_id] + 1;
    
    // Elimino la task de la máquina origen.
    if (s->__machine_assignment_count[machine_origin_id] > 1) {
        int ready = 0;
        for (int i = 0; (i < s->etc_matrix->tasks_count) 
            && (s->__machine_assignment[machine_origin_id][i] != MACHINE__EOT)
            && (ready == 0); i++) {
            
            if (s->__machine_assignment[machine_origin_id][i] == task_id) {
                s->__machine_assignment[machine_origin_id][i] = MACHINE__EMPTY;                
                ready = 1;
            }
        }
    } else {
        s->__machine_assignment[machine_origin_id][0] = MACHINE__EOT; 
    }
    
    s->__machine_assignment_count[machine_id] = s->__machine_assignment_count[machine_origin_id] - 1;
    
    if (recompute_makespan == 1) refresh_makespan(s);
}

void swap_tasks(struct solution *s, int task_a_id, int task_b_id) {
    assert(task_a_id < s->etc_matrix->tasks_count);
    assert(task_a_id >= 0);
    assert(task_b_id < s->etc_matrix->tasks_count);
    assert(task_b_id >= 0);
    assert(s->__task_assignment[task_a_id] != TASK__NOT_ASSIGNED);
    assert(s->__task_assignment[task_b_id] != TASK__NOT_ASSIGNED);
    
    int recompute_makespan = 0;    
    int machine_a_id = s->__task_assignment[task_a_id];
    int machine_b_id = s->__task_assignment[task_b_id];
    
    float machine_a_ct = s->__machine_compute_time[machine_a_id];
    float machine_b_ct = s->__machine_compute_time[machine_b_id];
    
    s->__machine_compute_time[machine_a_id] += get_etc_value(s->etc_matrix, machine_a_id, task_b_id);
    s->__machine_compute_time[machine_a_id] -= get_etc_value(s->etc_matrix, machine_a_id, task_a_id);
    s->__machine_compute_time[machine_b_id] += get_etc_value(s->etc_matrix, machine_b_id, task_a_id);
    s->__machine_compute_time[machine_b_id] -= get_etc_value(s->etc_matrix, machine_b_id, task_b_id);
    
    float machine_a_ct_new = s->__machine_compute_time[machine_a_id];
    float machine_b_ct_new = s->__machine_compute_time[machine_b_id];    
    
    if (machine_a_ct_new > s->__makespan || machine_b_ct_new > s->__makespan) {
        if (machine_a_ct_new > machine_b_ct_new) {
            s->__makespan = machine_a_ct_new;
        } else {
            s->__makespan = machine_b_ct_new;
        }
    } else {
        if (machine_a_ct == s->__makespan || machine_b_ct == s->__makespan) {
            recompute_makespan = 1;
        }
    }

    s->__task_assignment[task_a_id] = machine_b_id;
    s->__task_assignment[task_b_id] = machine_a_id;
    
    // Agrego la task A a la máquina B.
    int ready = 0;
    for (int i = 0; (i < s->etc_matrix->tasks_count) 
        && (s->__machine_assignment[machine_b_id][i] != MACHINE__EOT)
        && (ready == 0); i++) {
        
        if (s->__machine_assignment[machine_b_id][i] == task_b_id) {
            s->__machine_assignment[machine_b_id][i] = task_a_id;
            ready = 1;
        }
    }
    
    // Agrego la task B a la máquina A.
    ready = 0;
    for (int i = 0; (i < s->etc_matrix->tasks_count) 
        && (s->__machine_assignment[machine_a_id][i] != MACHINE__EOT)
        && (ready == 0); i++) {
        
        if (s->__machine_assignment[machine_a_id][i] == task_a_id) {
            s->__machine_assignment[machine_a_id][i] = task_b_id;                
            ready = 1;
        }
    }
    
    if (recompute_makespan == 1) refresh_makespan(s);
}

int get_task_assignment(struct solution *s, int task_id) {
    assert(task_id < s->etc_matrix->tasks_count);
    assert(task_id >= 0);
    
    return s->__task_assignment[task_id];
}

float get_machine_compute_time(struct solution *s, int machine_id) {
    assert(machine_id < s->etc_matrix->machines_count);
    assert(machine_id >= 0);

    return s->__machine_compute_time[machine_id];
}

void refresh_makespan(struct solution *s) {
    s->__makespan = s->__machine_compute_time[0];
    
    for (int i = 1; i < s->etc_matrix->machines_count; i++) {
        if (s->__machine_compute_time[i] > s->__makespan) {
            s->__makespan = s->__machine_compute_time[i];
        }
    }
}

int get_task_in_machine(struct solution *s, int position) {

}

int* get_all_tasks(struct solution *s, int position) {

}

float get_makespan(struct solution *s) {
    return s->__makespan;
}

void validate_solution(struct solution *s) {
	fprintf(stdout, "[INFO] Validate solution =========================== \n");
	fprintf(stdout, "[INFO] Dimension (%d x %d).\n", s->etc_matrix->tasks_count, s->etc_matrix->machines_count);
	fprintf(stdout, "[INFO] Makespan: %f.\n", s->__makespan);
	
	{
		float aux_makespan = 0.0;

		for (int machine = 0; machine < s->etc_matrix->machines_count; machine++) {
			if (aux_makespan < s->__machine_compute_time[machine]) {
				aux_makespan = s->__machine_compute_time[machine];
			}
		}
	
		assert(s->__makespan == aux_makespan);
	}
	
	for (int machine = 0; machine < s->etc_matrix->machines_count; machine++) {
		float aux_compute_time;
		aux_compute_time = 0.0;
	
		int assigned_tasks_count = 0;
	
		for (int task = 0; task < s->etc_matrix->tasks_count; task++) {
			if (s->__task_assignment[task] == machine) {
				aux_compute_time += get_etc_value(s->etc_matrix, machine, task);
				assigned_tasks_count++;
			}
		}

		/*if (DEBUG) {
			fprintf(stdout, "[DEBUG] Machine %d >> assigned tasks %d, compute time %f, expected compute time %f.\n",
				machine, assigned_tasks_count, aux_compute_time, s->__machine_compute_time[machine]);
		}*/
		
		assert(s->__machine_compute_time[machine] == aux_compute_time);
	}
	
	for (int task = 0; task < s->etc_matrix->tasks_count; task++) {
		assert(s->__task_assignment[task] >= 0);
		assert(s->__task_assignment[task] < s->etc_matrix->machines_count);
	}
	
	fprintf(stdout, "[INFO] The current solution is valid.\n");
	fprintf(stdout, "[INFO] ============================================= \n");	
}

void show_solution(struct solution *s) {
	fprintf(stdout, "[INFO] Show solution =========================== \n");
	fprintf(stdout, "[INFO] Dimension (%d x %d).\n", s->etc_matrix->tasks_count, s->etc_matrix->machines_count);

	fprintf(stdout, "   Makespan: %f.\n", s->__makespan);
	
	for (int machine = 0; machine < s->etc_matrix->machines_count; machine++) {
		fprintf(stdout, "   Machine: %d -> execution time: %f.\n", machine, s->__machine_compute_time[machine]);
	}
	
	for (int task = 0; task < s->etc_matrix->tasks_count; task++) {
		fprintf(stdout, "   Task: %d -> assigned to: %d.\n", task, s->__task_assignment[task]);
	}
	fprintf(stdout, "[INFO] ========================================= \n");
}
