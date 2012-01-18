#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "config.h"
#include "solution.h"

struct solution* create_empty_solution(struct etc_matrix *etc) {
	struct solution *new_solution;
	new_solution = (struct solution*)(malloc(sizeof(struct solution)));
	
	if (new_solution == NULL) {
		fprintf(stderr, "[ERROR] Solicitando memoria para new_solution.\n");
		exit(EXIT_FAILURE);
	}

    init_empty_solution(etc, new_solution);
	
	return new_solution;
}

void init_empty_solution(struct etc_matrix *etc, struct solution *new_solution) {
    new_solution->etc = etc;
	
	//=== Estructura orientada a tareas.
	new_solution->__task_assignment = (int*)(malloc(sizeof(int) * etc->tasks_count));
	
	if (new_solution->__task_assignment == NULL) {
		fprintf(stderr, "[ERROR] Solicitando memoria para el new_solution->task_assignment.\n");
		exit(EXIT_FAILURE);
	}
	
	for (int task = 0; task < etc->tasks_count; task++) {
		new_solution->__task_assignment[task] = TASK__NOT_ASSIGNED; /* not yet assigned */
	}
	
	//=== Estructura orientada a máquinas.
	new_solution->__machine_assignment = (int**)(malloc(sizeof(int*) * etc->machines_count));
	
	if (new_solution->__machine_assignment == NULL) {
		fprintf(stderr, "[ERROR] Solicitando memoria para el new_solution->machine_assignment.\n");
		exit(EXIT_FAILURE);
	}
	
	for (int machine = 0; machine < etc->machines_count; machine++) {
		new_solution->__machine_assignment[machine] = (int*)(malloc(sizeof(int) * etc->tasks_count));
		
		if (new_solution->__machine_assignment[machine] == NULL) {
		    fprintf(stderr, "[ERROR] Solicitando memoria para el new_solution->__machine_assignment[%d].\n", machine);
		    exit(EXIT_FAILURE);
	    }
	}	
	
	new_solution->__machine_assignment_count = (int*)(malloc(sizeof(int) * etc->machines_count));
	memset(new_solution->__machine_assignment_count, 0, sizeof(int) * etc->machines_count);
	
	//=== Estructura de machine compute time.
	new_solution->__makespan = 0.0;
	new_solution->__worst_ct_machine_id = -1;
	new_solution->__machine_compute_time = (float*)(malloc(sizeof(float) * etc->machines_count));
	
	if (new_solution->__machine_compute_time == NULL) {
		fprintf(stderr, "[ERROR] Solicitando memoria para el new_solution->__machine_compute_time.\n");
		exit(EXIT_FAILURE);
	}
	
	for (int machine = 0; machine < etc->machines_count; machine++) {
		new_solution->__machine_compute_time[machine] = 0.0;
	}
	
	//=== Estructura de energy.
	new_solution->__total_energy_consumption = 0.0;
	new_solution->__worst_energy_machine_id = -1;
	new_solution->__machine_energy_consumption = (float*)(malloc(sizeof(float) * etc->machines_count));
	
	if (new_solution->__machine_energy_consumption == NULL) {
		fprintf(stderr, "[ERROR] Solicitando memoria para el new_solution->__machine_energy_consumption.\n");
		exit(EXIT_FAILURE);
	}
	
	for (int machine = 0; machine < etc->machines_count; machine++) {
		new_solution->__machine_energy_consumption[machine] = 0.0;
	}
}

void clone_solution(struct solution *dst, struct solution *src) {
	dst->__makespan = src->__makespan;
	memcpy(dst->__task_assignment, src->__task_assignment, sizeof(int) * src->etc->tasks_count);
	memcpy(dst->__machine_compute_time, src->__machine_compute_time, sizeof(float) * src->etc->machines_count);
}

void free_solution(struct solution *s) {
	free(s->__task_assignment);
	free(s->__machine_assignment_count);
	for (int machine = 0; machine < s->etc->machines_count; machine++) {
		free(s->__machine_assignment[machine]);
	}
	free(s->__machine_assignment);
	
	free(s->__machine_compute_time);
	free(s->__machine_energy_consumption);
}

void assign_task_to_machine(struct solution *s, int machine_id, int task_id) {
    assert(machine_id < s->etc->machines_count);
    assert(machine_id >= 0);
    assert(task_id < s->etc->tasks_count);
    assert(task_id >= 0);
    assert(s->__task_assignment[task_id] == TASK__NOT_ASSIGNED);
       
    s->__machine_compute_time[machine_id] += get_etc_value(s->etc, machine_id, task_id);
    if (s->__machine_compute_time[machine_id] > s->__makespan) s->__makespan = s->__machine_compute_time[machine_id];
    
    s->__task_assignment[task_id] = machine_id;

	int current_machine_task_count = s->__machine_assignment_count[machine_id];
	s->__machine_assignment[machine_id][current_machine_task_count] = task_id;
	s->__machine_assignment_count[machine_id] = current_machine_task_count + 1;
}

void move_task_to_machine(struct solution *s, int task_id, int machine_id) {
    assert(machine_id < s->etc->machines_count);
    assert(machine_id >= 0);
    assert(task_id < s->etc->tasks_count);
    assert(task_id >= 0);
    assert(s->__task_assignment[task_id] != TASK__NOT_ASSIGNED);
    assert(s->__task_assignment[task_id] != machine_id);
    
    int recompute_makespan = 0;    
    int machine_origin_id = s->__task_assignment[task_id];
    
    // Actualizo los compute time.
    s->__machine_compute_time[machine_id] += get_etc_value(s->etc, machine_id, task_id);
    
    if (s->__machine_compute_time[machine_id] > s->__makespan) {
        s->__makespan = s->__machine_compute_time[machine_id];
    } else if (s->__machine_compute_time[machine_origin_id] == s->__makespan) {
        recompute_makespan = 1;
    }
    
    s->__machine_compute_time[machine_origin_id] -= get_etc_value(s->etc, machine_origin_id, task_id);
    
    // Quito la task a la máquina origen.
    int current_machine_origin_task_count = get_machine_tasks_count(s, machine_origin_id);
    int machine_origin_task_pos = get_machine_task_pos(s, machine_origin_id, task_id);
    
    if (machine_origin_task_pos + 1 == current_machine_origin_task_count) {
        // Es la última...
    } else {
        s->__machine_assignment[machine_id][machine_origin_task_pos] = s->__machine_assignment[machine_id][current_machine_origin_task_count-1];
    }

    s->__machine_assignment_count[machine_id] = current_machine_origin_task_count - 1;
     
    // Agrego la task a la máquina destino.
    s->__task_assignment[task_id] = machine_id;
    
    int current_machine_task_count = s->__machine_assignment_count[machine_id];
	s->__machine_assignment[machine_id][current_machine_task_count] = task_id;
	s->__machine_assignment_count[machine_id] = current_machine_task_count + 1;   
    
    // Si es necesario recalculo el makespan de la solución.
    if (recompute_makespan == 1) refresh_makespan(s);
}

void swap_tasks(struct solution *s, int task_a_id, int task_b_id) {
    assert(task_a_id < s->etc->tasks_count);
    assert(task_a_id >= 0);
    assert(task_b_id < s->etc->tasks_count);
    assert(task_b_id >= 0);
    assert(s->__task_assignment[task_a_id] != TASK__NOT_ASSIGNED);
    assert(s->__task_assignment[task_b_id] != TASK__NOT_ASSIGNED);
    
    int recompute_makespan = 0;    
    int machine_a_id = s->__task_assignment[task_a_id];
    int machine_b_id = s->__task_assignment[task_b_id];
    
    float machine_a_ct = s->__machine_compute_time[machine_a_id];
    float machine_b_ct = s->__machine_compute_time[machine_b_id];
    
    s->__machine_compute_time[machine_a_id] += get_etc_value(s->etc, machine_a_id, task_b_id);
    s->__machine_compute_time[machine_a_id] -= get_etc_value(s->etc, machine_a_id, task_a_id);
    s->__machine_compute_time[machine_b_id] += get_etc_value(s->etc, machine_b_id, task_a_id);
    s->__machine_compute_time[machine_b_id] -= get_etc_value(s->etc, machine_b_id, task_b_id);
    
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

    int task_a_pos, task_b_pos;
    task_a_pos = get_machine_task_pos(s, machine_a_id, task_a_id);;
    task_b_pos = get_machine_task_pos(s, machine_b_id, task_b_id);;

    s->__machine_assignment[machine_a_id][task_a_pos] = task_b_id;
    s->__machine_assignment[machine_b_id][task_b_pos] = task_a_id;

    s->__task_assignment[task_a_id] = machine_b_id;
    s->__task_assignment[task_b_id] = machine_a_id;

    // Si es necesario recalculo el makespan.    
    if (recompute_makespan == 1) refresh_makespan(s);
}

float get_machine_compute_time(struct solution *s, int machine_id) {
    assert(machine_id < s->etc->machines_count);
    assert(machine_id >= 0);

    return s->__machine_compute_time[machine_id];
}

void refresh_makespan(struct solution *s) {
    s->__makespan = s->__machine_compute_time[0];
    
    for (int i = 1; i < s->etc->machines_count; i++) {
        if (s->__machine_compute_time[i] > s->__makespan) {
            s->__makespan = s->__machine_compute_time[i];
        }
    }
}

int get_task_assigned_machine_id(struct solution *s, int task_id) {
    assert(task_id < s->etc->tasks_count);
    assert(task_id >= 0);
    
    return s->__task_assignment[task_id];
}

int get_machine_tasks_count(struct solution *s, int machine_id) {
    assert(machine_id < s->etc->machines_count);
    assert(machine_id >= 0);

    return s->__machine_assignment_count[machine_id];
}

int get_machine_task_id(struct solution *s, int machine_id, int task_position) {
    assert(machine_id < s->etc->machines_count);
    assert(machine_id >= 0);
    assert(task_position >= 0);
    assert(s->__machine_assignment_count[machine_id] < task_position);
    
    return s->__machine_assignment[machine_id][task_position];
}

int get_machine_task_pos(struct solution *s, int machine_id, int task_id) {
    assert(machine_id < s->etc->machines_count);
    assert(machine_id >= 0);
    assert(task_id < s->etc->tasks_count);
    assert(task_id >= 0);
    assert(s->__task_assignment[task_id] == machine_id);

    int i = 0;

    if (s->__machine_assignment_count[machine_id] > 0) {
        for (; (i < s->__machine_assignment_count[machine_id]) 
            && (s->__machine_assignment[machine_id][i] != task_id); i++) {
        }
        
        if (i < s->__machine_assignment_count[machine_id]) {
            return i;
        } else {
            return -1;
        }
    } else {
        return -1;
    }
}

float get_makespan(struct solution *s) {
    return s->__makespan;
}

void validate_solution(struct solution *s) {
	fprintf(stdout, "[INFO] Validate solution =========================== \n");
	fprintf(stdout, "[INFO] Dimension (%d x %d).\n", s->etc->tasks_count, s->etc->machines_count);
	fprintf(stdout, "[INFO] Makespan: %f.\n", s->__makespan);
	
	{
		float aux_makespan = 0.0;

		for (int machine = 0; machine < s->etc->machines_count; machine++) {
			if (aux_makespan < s->__machine_compute_time[machine]) {
				aux_makespan = s->__machine_compute_time[machine];
			}
		}
	
		assert(s->__makespan == aux_makespan);
	}
	
	for (int machine = 0; machine < s->etc->machines_count; machine++) {
		float aux_compute_time;
		aux_compute_time = 0.0;
	
		int assigned_tasks_count = 0;
	
		for (int task = 0; task < s->etc->tasks_count; task++) {
			if (s->__task_assignment[task] == machine) {
				aux_compute_time += get_etc_value(s->etc, machine, task);
				assigned_tasks_count++;
			}
		}

		/*if (DEBUG) {
			fprintf(stdout, "[DEBUG] Machine %d >> assigned tasks %d, compute time %f, expected compute time %f.\n",
				machine, assigned_tasks_count, aux_compute_time, s->__machine_compute_time[machine]);
		}*/
		
		assert(s->__machine_compute_time[machine] == aux_compute_time);
	}
	
	for (int task = 0; task < s->etc->tasks_count; task++) {
		assert(s->__task_assignment[task] >= 0);
		assert(s->__task_assignment[task] < s->etc->machines_count);
	}
	
	fprintf(stdout, "[INFO] The current solution is valid.\n");
	fprintf(stdout, "[INFO] ============================================= \n");	
}

void show_solution(struct solution *s) {
	fprintf(stdout, "[INFO] Show solution =========================== \n");
	fprintf(stdout, "[INFO] Dimension (%d x %d).\n", s->etc->tasks_count, s->etc->machines_count);

	fprintf(stdout, "   Makespan: %f.\n", s->__makespan);
	
	for (int machine = 0; machine < s->etc->machines_count; machine++) {
		fprintf(stdout, "   Machine: %d -> execution time: %f.\n", machine, s->__machine_compute_time[machine]);
	}
	
	for (int task = 0; task < s->etc->tasks_count; task++) {
		fprintf(stdout, "   Task: %d -> assigned to: %d.\n", task, s->__task_assignment[task]);
	}
	fprintf(stdout, "[INFO] ========================================= \n");
}
