#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../../random/cpu_rand.h"
#include "../../random/cpu_drand48.h"
#include "../../random/cpu_mt.h"

#include "machine_sel_complex.h"

inline void rand_generate(pals_cpu_1pop_thread_arg *thread_instance, double &random) {
    #ifdef CPU_MERSENNE_TWISTER
    random = cpu_mt_generate(*(thread_instance->thread_random_state));
    #endif
    #ifdef CPU_RAND
    random = cpu_rand_generate(*(thread_instance->thread_random_state));
    #endif
    #ifdef CPU_DRAND48
    random = cpu_drand48_generate(*(thread_instance->thread_random_state));
    #endif
}

void machines_complex_selection(pals_cpu_1pop_thread_arg *thread_instance, solution *selected_solution,
    int search_type, int &machine_a, int &machine_b)
{
    double random;

    if (search_type == PALS_CPU_1POP_SEARCH__MAKESPAN_GREEDY)
    {
        //fprintf(stdout, "[DEBUG] Makespan greedy\n");

        rand_generate(thread_instance, random);

        if (random > PALS_CPU_1POP_SEARCH__MAKESPAN_GREEDY_PSEL_WORST)
        {
            rand_generate(thread_instance, random);
            machine_a = (int)floor(random * thread_instance->etc->machines_count);

            // fprintf(stdout, "[DEBUG] Random machine_a = %d\n", machine_a);
        }
        else
        {
            machine_a = get_worst_ct_machine_id(selected_solution);

            //fprintf(stdout, "[DEBUG] Worst CT machine_a = %d\n", machine_a);
        }

        rand_generate(thread_instance, random);

        if (random > PALS_CPU_1POP_SEARCH__MAKESPAN_GREEDY_PSEL_BEST)
        {
            rand_generate(thread_instance, random);

            // Siempre selecciono la segunda mquina aleatoriamente.
            machine_b = (int)floor(random * (thread_instance->etc->machines_count - 1));

            //fprintf(stdout, "[DEBUG] Random machine_b = %d\n", machine_b);
        }
        else
        {
            machine_b = get_best_ct_machine_id(selected_solution);

            //fprintf(stdout, "[DEBUG] Best CT machine_b = %d\n", machine_b);
        }
    }
    else if (search_type == PALS_CPU_1POP_SEARCH__ENERGY_GREEDY)
    {
        //fprintf(stdout, "[DEBUG] Energy greedy\n");

        rand_generate(thread_instance, random);

        if (random > PALS_CPU_1POP_SEARCH__ENERGY_GREEDY_PSEL_WORST)
        {
            rand_generate(thread_instance, random);
            machine_a = (int)floor(random * thread_instance->etc->machines_count);

            //fprintf(stdout, "[DEBUG] Random machine_a = %d\n", machine_a);
        }
        else
        {
            machine_a = get_worst_energy_machine_id(selected_solution);

            //fprintf(stdout, "[DEBUG] Worst energy machine_a = %d\n", machine_a);
        }

        rand_generate(thread_instance, random);

        if (random > PALS_CPU_1POP_SEARCH__ENERGY_GREEDY_PSEL_BEST)
        {
            rand_generate(thread_instance, random);

            // Siempre selecciono la segunda mquina aleatoriamente.
            machine_b = (int)floor(random * (thread_instance->etc->machines_count - 1));

            //fprintf(stdout, "[DEBUG] Random machine_b = %d\n", machine_b);
        }
        else
        {
            machine_b = get_best_energy_machine_id(selected_solution);

            //fprintf(stdout, "[DEBUG] Worst energy machine_b = %d\n", machine_b);
        }
    }
    else
    {
        rand_generate(thread_instance, random);

        // La estrategia es aleatoria.
        machine_a = (int)floor(random * thread_instance->etc->machines_count);

        rand_generate(thread_instance, random);

        // Siempre selecciono la segunda mquina aleatoriamente.
        machine_b = (int)floor(random * (thread_instance->etc->machines_count - 1));
    }

    int machine_a_task_count = get_machine_tasks_count(selected_solution, machine_a);
    while (machine_a_task_count == 0)
    {
        machine_a = (machine_a + 1) % thread_instance->etc->machines_count;
        machine_a_task_count = get_machine_tasks_count(selected_solution, machine_a);
    }

    if (machine_a == machine_b) machine_b = (machine_b + 1) % thread_instance->etc->machines_count;
    int machine_b_task_count = get_machine_tasks_count(selected_solution, machine_b);
}
