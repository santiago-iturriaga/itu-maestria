#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "config.h"
#include "solution.h"

struct solution* create_empty_solution(struct etc_matrix *etc, struct energy_matrix *energy) {
	struct solution *new_solution;
	new_solution = (struct solution*)(malloc(sizeof(struct solution)));
	
	if (new_solution == NULL) {
		fprintf(stderr, "[ERROR] Solicitando memoria para new_solution.\n");
		exit(EXIT_FAILURE);
	}

    init_empty_solution(etc, energy, new_solution);
	
	return new_solution;
}

void init_empty_solution(struct etc_matrix *etc, struct energy_matrix *energy, struct solution *new_solution) {
    new_solution->initialized = 1;

    new_solution->etc = etc;
    new_solution->energy = energy;
	
	//=== Estructura orientada a tareas.
	new_solution->__task_assignment = (int*)(malloc(sizeof(int) * etc->tasks_count));
	
	if (new_solution->__task_assignment == NULL) {
		fprintf(stderr, "[ERROR] Solicitando memoria para el new_solution->task_assignment.\n");
		exit(EXIT_FAILURE);
	}
	
	for (int task = 0; task < etc->tasks_count; task++) {
		new_solution->__task_assignment[task] = SOLUTION__TASK_NOT_ASSIGNED; /* not yet assigned */
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

void clone_solution(struct solution *dst, struct solution *src, int clone_status) {
    if (clone_status == 1) {
        dst->status = src->status;
    }

	memcpy(dst->__task_assignment, src->__task_assignment, sizeof(int) * src->etc->tasks_count);
	for (int machine = 0; machine < src->etc->machines_count; machine++) {
		memcpy(dst->__machine_assignment[machine], src->__machine_assignment[machine], sizeof(int) * src->etc->tasks_count);
	}
   	memcpy(dst->__machine_assignment_count, src->__machine_assignment_count, sizeof(int) * src->etc->machines_count);
    
    // Makespan
	memcpy(dst->__machine_compute_time, src->__machine_compute_time, sizeof(float) * src->etc->machines_count);
	dst->__worst_ct_machine_id = src->__worst_ct_machine_id;
	dst->__makespan = src->__makespan;
	
	// Energy
	memcpy(dst->__machine_energy_consumption, src->__machine_energy_consumption, sizeof(float) * src->etc->machines_count);
	dst->__worst_energy_machine_id = src->__worst_energy_machine_id;
	dst->__total_energy_consumption = src->__total_energy_consumption;
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
    assert(s->__task_assignment[task_id] == SOLUTION__TASK_NOT_ASSIGNED);
       
    // Actualizo el makespan.
    s->__machine_compute_time[machine_id] += get_etc_value(s->etc, machine_id, task_id);
        
    // Asigno la tarea.
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
    assert(s->__task_assignment[task_id] != SOLUTION__TASK_NOT_ASSIGNED);
    assert(s->__task_assignment[task_id] != machine_id);
    
    int recompute_makespan = 0;
    int recompute_energy = 0;
    
    int machine_origin_id = s->__task_assignment[task_id];
    
    // Actualizo los compute time.
    s->__machine_compute_time[machine_id] += get_etc_value(s->etc, machine_id, task_id);
    
    if (s->__machine_compute_time[machine_id] > s->__makespan) {
        s->__makespan = s->__machine_compute_time[machine_id];
        s->__worst_ct_machine_id = machine_id;
        
        recompute_energy = 1;
    } else if (s->__machine_compute_time[machine_origin_id] == s->__makespan) {
        recompute_makespan = 1;
    }
    
    if (s->__machine_compute_time[machine_origin_id] < s->__makespan) {
        s->__machine_compute_time[machine_origin_id] -= get_etc_value(s->etc, machine_origin_id, task_id);
    } else {
        recompute_makespan = 1;
    }
    
    // Actualizo la energía.
    if (recompute_energy == 0) {
        if (recompute_makespan == 0) {
            // Máquina destino.
            float old_machine_energy = s->__machine_energy_consumption[machine_id];
            float new_machine_energy = (s->__machine_compute_time[machine_id] * get_energy_max_value(s->energy, machine_id))
                + ((s->__makespan - s->__machine_compute_time[machine_id]) * get_energy_idle_value(s->energy, machine_id));
                
            s->__machine_energy_consumption[machine_id] = new_machine_energy;
            
            if (machine_id == s->__worst_energy_machine_id) {
                // Sigue siendo la peor máquina.    
            } else if (new_machine_energy > s->__machine_energy_consumption[s->__worst_energy_machine_id]) {
                s->__worst_energy_machine_id = machine_id;
            }
            
            s->__total_energy_consumption = s->__total_energy_consumption - old_machine_energy + new_machine_energy;
            
            // Máquina origen.
            old_machine_energy = s->__machine_energy_consumption[machine_origin_id];
            new_machine_energy = (s->__machine_compute_time[machine_origin_id] * get_energy_max_value(s->energy, machine_origin_id))
                + ((s->__makespan - s->__machine_compute_time[machine_origin_id]) * get_energy_idle_value(s->energy, machine_origin_id));
                
            s->__machine_energy_consumption[machine_origin_id] = new_machine_energy;
            
            if (machine_origin_id == s->__worst_energy_machine_id) {
                // Busco el nuevo peor.
                refresh_worst_energy(s);
            }
            
            s->__total_energy_consumption = s->__total_energy_consumption - old_machine_energy + new_machine_energy;
        } else {
            recompute_energy = 1;
        }
    }
        
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
    
    // Si es necesario recalculo el makespan o la energía de la solución.
    if (recompute_makespan == 1) refresh_makespan(s);
    if (recompute_energy == 1) refresh_energy(s);
}

void move_task_to_machine_by_pos(struct solution *s, int machine_src, int task_src_pos, int machine_dst) {
    assert(machine_src != machine_dst);
    assert(machine_src < s->etc->machines_count);
    assert(machine_src >= 0);
    assert(machine_dst < s->etc->machines_count);
    assert(machine_dst >= 0);
    assert(task_src_pos >= 0);
    assert(s->__machine_assignment_count[machine_src] > task_src_pos);
    
    if (DEBUG_DEV) {
        fprintf(stdout, "[DEBUG] Move task sol. %ld: from (%d, %d=%d) to (%d)\n", 
            (long)s, machine_src, task_src_pos, s->__machine_assignment[machine_src][task_src_pos], machine_dst);
        
        /*
        fprintf(stdout, "        Maquina %d >> ", machine_src);
        for (int i = 0; i < s->__machine_assignment_count[machine_src]; i++) {
            fprintf(stdout, "%d ", s->__machine_assignment[machine_src][i]);
        }
        fprintf(stdout, "\n        Maquina %d >> ", machine_dst);
        for (int i = 0; i < s->__machine_assignment_count[machine_dst]; i++) {
            fprintf(stdout, "%d ", s->__machine_assignment[machine_dst][i]);
        }
        fprintf(stdout, "\n");*/
    }
    
    int recompute_makespan = 0;
    int recompute_energy = 0;
    
    int task_src_id;
    task_src_id = get_machine_task_id(s, machine_src, task_src_pos);
     
    assert(s->__task_assignment[task_src_id] != SOLUTION__TASK_NOT_ASSIGNED);
    assert(s->__task_assignment[task_src_id] != machine_dst);
    
    // Actualizo los compute time.
    s->__machine_compute_time[machine_dst] += get_etc_value(s->etc, machine_dst, task_src_id);
    
    if (s->__machine_compute_time[machine_dst] > s->__makespan) {
        s->__makespan = s->__machine_compute_time[machine_dst];
        s->__worst_ct_machine_id = machine_dst;
        
        recompute_energy = 1;
    } else if (s->__machine_compute_time[machine_src] == s->__makespan) {
        recompute_makespan = 1;
    }
    
    if (s->__machine_compute_time[machine_src] < s->__makespan) {
        s->__machine_compute_time[machine_src] -= get_etc_value(s->etc, machine_src, task_src_id);
    } else {
        recompute_makespan = 1;
    }
    
    // Actualizo la energía.
    if (recompute_energy == 0) {
        if (recompute_makespan == 0) {
            // Máquina destino.
            float old_machine_energy = s->__machine_energy_consumption[machine_dst];
            float new_machine_energy = (s->__machine_compute_time[machine_dst] * get_energy_max_value(s->energy, machine_dst))
                + ((s->__makespan - s->__machine_compute_time[machine_dst]) * get_energy_idle_value(s->energy, machine_dst));
                
            s->__machine_energy_consumption[machine_dst] = new_machine_energy;
            
            if (machine_dst == s->__worst_energy_machine_id) {
                // Sigue siendo la peor máquina.    
            } else if (new_machine_energy > s->__machine_energy_consumption[s->__worst_energy_machine_id]) {
                s->__worst_energy_machine_id = machine_dst;
            }
            
            s->__total_energy_consumption = s->__total_energy_consumption - old_machine_energy + new_machine_energy;
                       
            // Máquina origen.
            old_machine_energy = s->__machine_energy_consumption[machine_src];
            new_machine_energy = (s->__machine_compute_time[machine_src] * get_energy_max_value(s->energy, machine_src))
                + ((s->__makespan - s->__machine_compute_time[machine_src]) * get_energy_idle_value(s->energy, machine_src));
                
            s->__machine_energy_consumption[machine_src] = new_machine_energy;
            
            if (machine_src == s->__worst_energy_machine_id) {
                // Busco el nuevo peor.
                refresh_worst_energy(s);
            }
            
            s->__total_energy_consumption = s->__total_energy_consumption - old_machine_energy + new_machine_energy;
            
        } else {
            recompute_energy = 1;
        }
    }
        
    // Quito la task a la máquina origen.
    int machine_src_task_count = s->__machine_assignment_count[machine_src];
    int machine_dst_task_count = s->__machine_assignment_count[machine_dst];
    
    if (task_src_pos + 1 == machine_src_task_count) {
        // Es la última...
    } else {
        s->__machine_assignment[machine_src][task_src_pos] = 
            s->__machine_assignment[machine_src][machine_src_task_count-1];
    }

    s->__machine_assignment_count[machine_src]--;
     
    // Agrego la task a la máquina destino.
    s->__task_assignment[task_src_id] = machine_dst;
    
    //int current_machine_task_count = s->__machine_assignment_count[machine_id];
	s->__machine_assignment[machine_dst][machine_dst_task_count] = task_src_id;
	s->__machine_assignment_count[machine_dst] = machine_dst_task_count + 1;   
    
    // Si es necesario recalculo el makespan o la energía de la solución.
    if (recompute_makespan == 1) refresh_makespan(s);
    if (recompute_energy == 1) refresh_energy(s);
}

void swap_tasks(struct solution *s, int task_a_id, int task_b_id) {
    assert(task_a_id < s->etc->tasks_count);
    assert(task_a_id >= 0);
    assert(task_b_id < s->etc->tasks_count);
    assert(task_b_id >= 0);
    assert(s->__task_assignment[task_a_id] != SOLUTION__TASK_NOT_ASSIGNED);
    assert(s->__task_assignment[task_b_id] != SOLUTION__TASK_NOT_ASSIGNED);
    
    int recompute_makespan = 0;
    int recompute_energy = 0;
    
    // Calculo el compute time y el makespan.
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
        
        recompute_energy = 1;
    } else if (machine_a_ct == s->__makespan || machine_b_ct == s->__makespan) {
        recompute_makespan = 1;
    }
    
    // Calculo la energía.
    if (recompute_energy == 0) {
        if (recompute_makespan == 0) {
            int refresh_worst = 0;
        
            // Máquina a.
            float old_machine_energy = s->__machine_energy_consumption[machine_a_id];
            float new_machine_energy = (s->__machine_compute_time[machine_a_id] * get_energy_max_value(s->energy, machine_a_id))
                + ((s->__makespan - s->__machine_compute_time[machine_a_id]) * get_energy_idle_value(s->energy, machine_a_id));
                
            s->__machine_energy_consumption[machine_a_id] = new_machine_energy;
            
            if (machine_a_id == s->__worst_energy_machine_id) {
                if (new_machine_energy > old_machine_energy) {
                    // Sigue siendo la peor máquina.
                } else {
                    refresh_worst = 1;
                }
            } else if (new_machine_energy > s->__machine_energy_consumption[s->__worst_energy_machine_id]) {
                s->__worst_energy_machine_id = machine_a_id;
            }
            
            s->__total_energy_consumption = s->__total_energy_consumption - old_machine_energy + new_machine_energy;
            
            // Máquina b.
            old_machine_energy = s->__machine_energy_consumption[machine_b_id];
            new_machine_energy = (s->__machine_compute_time[machine_b_id] * get_energy_max_value(s->energy, machine_b_id))
                + ((s->__makespan - s->__machine_compute_time[machine_b_id]) * get_energy_idle_value(s->energy, machine_b_id));
                
            s->__machine_energy_consumption[machine_b_id] = new_machine_energy;
            
            if (machine_b_id == s->__worst_energy_machine_id) {
                if (new_machine_energy > old_machine_energy) {
                    // Sigue siendo la peor máquina.
                } else {
                    refresh_worst = 1;
                }
            } else if (new_machine_energy > s->__machine_energy_consumption[s->__worst_energy_machine_id]) {
                s->__worst_energy_machine_id = machine_b_id;
            }
            
            s->__total_energy_consumption = s->__total_energy_consumption - old_machine_energy + new_machine_energy;
            
            if (refresh_worst == 1) {
                refresh_worst_energy(s);
            }
        } else {
            recompute_energy = 1;
        }
    }

    // Asigno las tareas.
    int task_a_pos, task_b_pos;
    task_a_pos = get_machine_task_pos(s, machine_a_id, task_a_id);;
    task_b_pos = get_machine_task_pos(s, machine_b_id, task_b_id);;

    s->__machine_assignment[machine_a_id][task_a_pos] = task_b_id;
    s->__machine_assignment[machine_b_id][task_b_pos] = task_a_id;

    s->__task_assignment[task_a_id] = machine_b_id;
    s->__task_assignment[task_b_id] = machine_a_id;

    // Si es necesario recalculo el makespan o la energía de la solución.
    if (recompute_makespan == 1) refresh_makespan(s);
    if (recompute_energy == 1) refresh_energy(s);
}

void swap_tasks_by_pos(struct solution *s, int machine_a, int task_a_pos, int machine_b, int task_b_pos) {   
    assert(machine_a != machine_b);    
    assert(machine_a < s->etc->machines_count);
    assert(machine_a >= 0);
    assert(machine_b < s->etc->machines_count);
    assert(machine_b >= 0);
    assert(task_a_pos >= 0);
    assert(task_b_pos >= 0);

    if (DEBUG_DEV) {
        fprintf(stdout, "[DEBUG] >>>> Pre Move task sol. %ld: (%d, %d=%d) with (%d, %d=%d)\n", (long)s, 
            machine_a, task_a_pos, s->__machine_assignment[machine_a][task_a_pos],
            machine_b, task_b_pos, s->__machine_assignment[machine_b][task_b_pos]);
              
        fprintf(stdout, "        Maquina %d (%d) (%f) >> ", 
            machine_a, s->__machine_assignment_count[machine_a], s->__machine_compute_time[machine_a]);
            
        for (int i = 0; i < s->__machine_assignment_count[machine_a]; i++) {
            fprintf(stdout, "%d ", s->__machine_assignment[machine_a][i]);
        }
        
        fprintf(stdout, "\n        Maquina %d (%d) (%f) >> ", 
            machine_b, s->__machine_assignment_count[machine_b], s->__machine_compute_time[machine_b]);
            
        for (int i = 0; i < s->__machine_assignment_count[machine_b]; i++) {
            fprintf(stdout, "%d ", s->__machine_assignment[machine_b][i]);
        }
        fprintf(stdout, "\n");
    }    

    assert(s->__machine_assignment_count[machine_a] > task_a_pos);
    assert(s->__machine_assignment_count[machine_b] > task_b_pos);

    int recompute_makespan = 0;
    int recompute_energy = 0;
        
    // Calculo el compute time y el makespan.
    int machine_a_id = machine_a;
    int machine_b_id = machine_b;

    int task_a_id = get_machine_task_id(s, machine_a, task_a_pos);
    int task_b_id = get_machine_task_id(s, machine_b, task_b_pos);
    
    assert(s->__task_assignment[task_a_id] != SOLUTION__TASK_NOT_ASSIGNED);
    assert(s->__task_assignment[task_b_id] != SOLUTION__TASK_NOT_ASSIGNED);
    
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
        
        recompute_energy = 1;
    } else if (machine_a_ct == s->__makespan || machine_b_ct == s->__makespan) {
        recompute_makespan = 1;
    }
    
    // Calculo la energía.
    if (recompute_energy == 0) {
        if (recompute_makespan == 0) {
            int refresh_worst = 0;
        
            // Máquina a.
            float old_machine_energy = s->__machine_energy_consumption[machine_a_id];
            float new_machine_energy = (s->__machine_compute_time[machine_a_id] * get_energy_max_value(s->energy, machine_a_id))
                + ((s->__makespan - s->__machine_compute_time[machine_a_id]) * get_energy_idle_value(s->energy, machine_a_id));
                
            s->__machine_energy_consumption[machine_a_id] = new_machine_energy;
            
            if (machine_a_id == s->__worst_energy_machine_id) {
                if (new_machine_energy > old_machine_energy) {
                    // Sigue siendo la peor máquina.
                } else {
                    refresh_worst = 1;
                }
            } else if (new_machine_energy > s->__machine_energy_consumption[s->__worst_energy_machine_id]) {
                s->__worst_energy_machine_id = machine_a_id;
            }
            
            s->__total_energy_consumption = s->__total_energy_consumption - old_machine_energy + new_machine_energy;
            
            // Máquina b.
            old_machine_energy = s->__machine_energy_consumption[machine_b_id];
            new_machine_energy = (s->__machine_compute_time[machine_b_id] * get_energy_max_value(s->energy, machine_b_id))
                + ((s->__makespan - s->__machine_compute_time[machine_b_id]) * get_energy_idle_value(s->energy, machine_b_id));
                
            s->__machine_energy_consumption[machine_b_id] = new_machine_energy;
            
            if (machine_b_id == s->__worst_energy_machine_id) {
                if (new_machine_energy > old_machine_energy) {
                    // Sigue siendo la peor máquina.
                } else {
                    refresh_worst = 1;
                }
            } else if (new_machine_energy > s->__machine_energy_consumption[s->__worst_energy_machine_id]) {
                s->__worst_energy_machine_id = machine_b_id;
            }
            
            s->__total_energy_consumption = s->__total_energy_consumption - old_machine_energy + new_machine_energy;
            
            if (refresh_worst == 1) {
                refresh_worst_energy(s);
            }
        } else {
            recompute_energy = 1;
        }
    }

    // Asigno las tareas.
    s->__machine_assignment[machine_a_id][task_a_pos] = task_b_id;
    s->__machine_assignment[machine_b_id][task_b_pos] = task_a_id;

    s->__task_assignment[task_a_id] = machine_b_id;
    s->__task_assignment[task_b_id] = machine_a_id;

    // Si es necesario recalculo el makespan o la energía de la solución.
    if (recompute_makespan == 1) refresh_makespan(s);
    if (recompute_energy == 1) refresh_energy(s);
    
    if (DEBUG_DEV) {
        fprintf(stdout, "[DEBUG] >>>> Post Move task sol. %ld: (%d, %d=%d) with (%d, %d=%d)\n", (long)s, 
            machine_a, task_a_pos, s->__machine_assignment[machine_a][task_a_pos],
            machine_b, task_b_pos, s->__machine_assignment[machine_b][task_b_pos]);
              
        fprintf(stdout, "        Maquina %d (%d) (%f) >> ", 
            machine_a, s->__machine_assignment_count[machine_a], s->__machine_compute_time[machine_a]);
            
        for (int i = 0; i < s->__machine_assignment_count[machine_a]; i++) {
            fprintf(stdout, "%d ", s->__machine_assignment[machine_a][i]);
        }
        
        fprintf(stdout, "\n        Maquina %d (%d) (%f) >> ", 
            machine_b, s->__machine_assignment_count[machine_b], s->__machine_compute_time[machine_b]);
            
        for (int i = 0; i < s->__machine_assignment_count[machine_b]; i++) {
            fprintf(stdout, "%d ", s->__machine_assignment[machine_b][i]);
        }
        fprintf(stdout, "\n");
    } 
}

float get_machine_compute_time(struct solution *s, int machine_id) {
    assert(machine_id < s->etc->machines_count);
    assert(machine_id >= 0);

    return s->__machine_compute_time[machine_id];
}

float get_makespan(struct solution *s) {
    return s->__makespan;
}

float get_energy(struct solution *s) {
    return s->__total_energy_consumption;
}

int get_worst_ct_machine_id(struct solution *s) {
    return s->__worst_ct_machine_id;
}

int get_worst_energy_machine_id(struct solution *s) {
    return s->__worst_energy_machine_id;
}

void refresh_energy(struct solution *s) {
    s->__worst_energy_machine_id = 0;
    s->__total_energy_consumption = 0.0;
    
    for (int i = 0; i < s->etc->machines_count; i++) {
        s->__machine_energy_consumption[i] = (s->__machine_compute_time[i] * get_energy_max_value(s->energy, i))
            + ((s->__makespan - s->__machine_compute_time[i]) * get_energy_idle_value(s->energy, i));
            
        s->__total_energy_consumption += s->__machine_energy_consumption[i];
        
        if (s->__machine_energy_consumption[i] > s->__machine_energy_consumption[s->__worst_energy_machine_id]) {
            s->__worst_energy_machine_id = i;
        }
    }
}

void refresh_worst_energy(struct solution *s) {
    s->__worst_energy_machine_id = 0;
    
    for (int i = 1; i < s->etc->machines_count; i++) {
        if (s->__machine_energy_consumption[i] > s->__machine_energy_consumption[s->__worst_energy_machine_id]) {
            s->__worst_energy_machine_id = i;
        }
    }
}

void refresh_makespan(struct solution *s) {
    s->__worst_ct_machine_id = 0;
    s->__makespan = s->__machine_compute_time[0];
    
    for (int i = 1; i < s->etc->machines_count; i++) {
        if (s->__machine_compute_time[i] > s->__makespan) {
            s->__worst_ct_machine_id = i;
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
        
    assert(s->__machine_assignment_count[machine_id] > task_position);
    
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
            fprintf(stdout, "[ERROR] La tarea %d no se encuentra en la máquina %d\n", task_id, machine_id);
            assert(0);
            //return -1;
        }
    } else {
        fprintf(stdout, "[ERROR] La tarea %d no se encuentra en la máquina %d\n", task_id, machine_id);
        assert(0);
        //return -1;
    }
}

void validate_solution(struct solution *s) {
	fprintf(stdout, "[INFO] Validate solution =========================== \n");
	fprintf(stdout, "[INFO] Dimension (%d x %d).\n", s->etc->tasks_count, s->etc->machines_count);
	fprintf(stdout, "[INFO] Makespan: %f.\n", s->__makespan);
	fprintf(stdout, "[INFO] Energy  : %f.\n", s->__total_energy_consumption);
	
	float current_makespan = s->__makespan;
	refresh_makespan(s);
	if (current_makespan != s->__makespan) {
	    fprintf(stdout, "[DEBUG] makespan %f, expected makespan %f.\n", current_makespan, s->__makespan);
	}
	assert(current_makespan == s->__makespan);	
	
	float current_energy = s->__total_energy_consumption;
	float *machine_energy = (float*)malloc(sizeof(float) * s->etc->machines_count);
	for (int i = 0; i < s->etc->machines_count; i++) {
	    machine_energy[i] = s->__machine_energy_consumption[i];
	}
	
	refresh_energy(s);
	
	float total_sum = 0;
	if (current_energy != s->__total_energy_consumption) {
	    fprintf(stdout, "[DEBUG] energy %f, expected energy %f.\n", current_energy, s->__total_energy_consumption);
	    
	    for (int i = 0; i < s->etc->machines_count; i++) {
	        if (machine_energy[i] != s->__machine_energy_consumption[i]) {
    	        fprintf(stdout, "        > machine %d > energy %f, expected energy %f.\n", i, machine_energy[i], s->__machine_energy_consumption[i]);
    	    }
    	    total_sum += machine_energy[i];
	    }
	}
	
	free(machine_energy);
	
	//assert(current_energy == s->__total_energy_consumption);
	
	for (int machine = 0; machine < s->etc->machines_count; machine++) {
		float aux_compute_time;
		aux_compute_time = 0.0;
	
		int assigned_tasks_count = 0;
	
		for (int task = 0; task < s->etc->tasks_count; task++) {
			if (s->__task_assignment[task] == machine) {
				aux_compute_time += get_etc_value(s->etc, machine, task);
				
				assert(get_machine_task_pos(s, machine, task) >= 0);
				
				assigned_tasks_count++;
			}
		}

		if (aux_compute_time != s->__machine_compute_time[machine]) {
			fprintf(stdout, "[DEBUG] Machine %d >> assigned tasks %d (%d), compute time %f, expected compute time %f.\n",
				machine, assigned_tasks_count, s->__machine_assignment_count[machine], aux_compute_time, s->__machine_compute_time[machine]);
		}
		
		//assert(s->__machine_compute_time[machine] == aux_compute_time);
		assert(s->__machine_assignment_count[machine] == assigned_tasks_count);
	}
	
	for (int task = 0; task < s->etc->tasks_count; task++) {
		assert(s->__task_assignment[task] >= 0);
		assert(s->__task_assignment[task] < s->etc->machines_count);
	}
	
	assert(s->__machine_compute_time[s->__worst_ct_machine_id] == s->__makespan);
	
	int current_worst_energy = s->__worst_ct_machine_id;
	refresh_worst_energy(s);
	assert(current_worst_energy == s->__worst_ct_machine_id);
	
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
