#include <stdio.h>
#include <stdlib.h>

#include "load_instance.h"

/* Reads the input files and stores de data in memory */
int load_instance(struct params *input, struct scenario *s, struct etc_matrix *etc, struct energy_matrix *energy) {
    #if defined(DEBUG_3)
        fprintf(stderr, "[DEBUG] load_instance\n");
    #endif
    
    FILE *scenario_file;

    if ((scenario_file = fopen(input->scenario_path, "r")) == NULL) {
        fprintf(stderr, "[ERROR] reading the scenario file %s.\n", input->scenario_path);
        return EXIT_FAILURE;
    }

    int cores;
    float ssj;
    float energy_idle;
    float energy_max;

    for (int machine = 0; machine < input->machines_count; machine++) {
        if (fscanf(scenario_file,"%d %f %f %f\n",&cores,&ssj,&energy_idle,&energy_max) != 4) {
            fprintf(stderr, "[ERROR] leyendo dato de scenario de la máquina %d.\n", machine);
            return EXIT_FAILURE;
        }

        set_scenario_machine(s, machine, cores, ssj, energy_idle, energy_max);
    }

    fclose(scenario_file);

    FILE *workload_file;

    if ((workload_file = fopen(input->workload_path, "r")) == NULL) {
        fprintf(stderr, "[ERROR] reading the workload file %s.\n", input->workload_path);
        return EXIT_FAILURE;
    }

    float value;
    for (int task = 0; task < input->tasks_count; task++) {
        for (int machine = 0; machine < input->machines_count; machine++) {
            if (fscanf(workload_file, "%f", &value) != 1) {
                fprintf(stderr, "[ERROR] leyendo dato de workload de la tarea %d para la máquina %d.\n", task, machine);
                return EXIT_FAILURE;
            }

            FLOAT task_etc;
            task_etc = value / get_scenario_ssj(s, machine);
            
            set_etc_value(etc, machine, task, task_etc);
            
            FLOAT idle_consumption;
            idle_consumption = task_etc * get_scenario_energy_idle(s, machine);
            FLOAT max_consumption;
            max_consumption = task_etc * get_scenario_energy_max(s, machine);            
            
            set_energy_value(energy, machine, task, max_consumption-idle_consumption);
        }
    }

    fclose(workload_file);

    return EXIT_SUCCESS;
}
