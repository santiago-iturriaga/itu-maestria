#include <stdio.h>
#include <stdlib.h>

#include "load_instance.h"
#include "global.h"
#include "energy_matrix.h"
#include "etc_matrix.h"
#include "scenario.h"

/* Reads the input files and stores de data in memory */
int load_instance() {
    #if defined(DEBUG_3)
        fprintf(stderr, "[DEBUG] load_instance\n");
    #endif
    
    FILE *scenario_file;

    if ((scenario_file = fopen(INPUT.scenario_path, "r")) == NULL) {
        fprintf(stderr, "[ERROR] reading the scenario file %s.\n", INPUT.scenario_path);
        return EXIT_FAILURE;
    }

    int cores;
    float ssj;
    float energy_idle;
    float energy_max;

    for (int machine = 0; machine < INPUT.machines_count; machine++) {
        if (fscanf(scenario_file,"%d %f %f %f\n",&cores,&ssj,&energy_idle,&energy_max) != 4) {
            fprintf(stderr, "[ERROR] leyendo dato de scenario de la máquina %d.\n", machine);
            return EXIT_FAILURE;
        }

        set_scenario_machine(machine, cores, ssj, energy_idle, energy_max);
    }

    fclose(scenario_file);

    FILE *workload_file;

    if ((workload_file = fopen(INPUT.workload_path, "r")) == NULL) {
        fprintf(stderr, "[ERROR] reading the workload file %s.\n", INPUT.workload_path);
        return EXIT_FAILURE;
    }

    FLOAT task_etc;
    FLOAT idle_consumption;
    FLOAT max_consumption;
    float value;

    for (int task = 0; task < INPUT.tasks_count; task++) {
        for (int machine = 0; machine < INPUT.machines_count; machine++) {
            if (fscanf(workload_file, "%f", &value) != 1) {
                fprintf(stderr, "[ERROR] leyendo dato de workload de la tarea %d para la máquina %d.\n", task, machine);
                return EXIT_FAILURE;
            }
            
            task_etc = value / get_scenario_ssj(machine);
            set_etc_value(machine, task, task_etc);
            
            idle_consumption = task_etc * get_scenario_energy_idle(machine);
            max_consumption = task_etc * get_scenario_energy_max(machine);            
            
            set_energy_value(machine, task, max_consumption-idle_consumption);
        }
    }

    fclose(workload_file);

    return EXIT_SUCCESS;
}
