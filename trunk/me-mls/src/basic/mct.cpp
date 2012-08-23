#include "../config.h"
#include "mct.h"

void compute_mct(struct solution *solution) {
    compute_custom_mct(solution, 0);
}

void compute_custom_mct(struct solution *solution, int starting_task) {
    int task;
    for (int index = 0; index < solution->etc->tasks_count; index++) {
        task = (starting_task + index) % solution->etc->tasks_count;
    
        int best_machine;
        best_machine = 0;
        
        float best_etc_value;
        best_etc_value = get_etc_value(solution->etc, 0, task);
    
        for (int machine = 1; machine < solution->etc->machines_count; machine++) {
            float etc_value;
            etc_value = get_etc_value(solution->etc, machine, task);
            
            if (get_machine_compute_time(solution, machine) + etc_value < 
                get_machine_compute_time(solution, best_machine) + best_etc_value) {
                
                best_etc_value = etc_value;
                best_machine = machine;
            }
        }
        
        assign_task_to_machine(solution, best_machine, task);
    }
    
    refresh_makespan(solution);
    refresh_energy(solution);
    
    #if defined(DEBUG) 
    fprintf(stdout, "[DEBUG] MCT Solution >> makespan: %f || energy: %f.\n", get_makespan(solution), get_energy(solution));
    #endif
    //else fprintf(stdout, "%f|%f\n", get_makespan(solution), get_energy(solution));
}
