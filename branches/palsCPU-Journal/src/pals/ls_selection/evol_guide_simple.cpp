#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../../random/cpu_rand.h"
#include "../../random/cpu_drand48.h"
#include "../../random/cpu_mt.h"

#include "evol_guide_simple.h"

void ls_best_swap_simple_selection(pals_cpu_1pop_thread_arg *thread_instance, solution *selected_solution,
    int search_type, int machine_a, int machine_b, int task_x_pos, int task_x_current,
    int task_y_pos, int task_y_current, float &best_delta_makespan, float &best_delta_energy,
    int &task_x_best_move_pos, int &machine_b_best_move_id, int &task_x_best_swap_pos, int &task_y_best_swap_pos)
{

    float machine_a_energy_idle = get_energy_idle_value(thread_instance->energy, machine_a);
    float machine_a_energy_max = get_energy_max_value(thread_instance->energy, machine_a);
    float machine_b_energy_idle = get_energy_idle_value(thread_instance->energy, machine_b);
    float machine_b_energy_max = get_energy_max_value(thread_instance->energy, machine_b);

    double machine_a_ct_old, machine_a_ct_new;
    double machine_b_ct_old, machine_b_ct_new;

    float current_makespan = get_makespan(selected_solution);
    //float current_energy = get_energy(selected_solution);

    // Maquina 1.
    machine_a_ct_old = get_machine_compute_time(selected_solution, machine_a);

    machine_a_ct_new = machine_a_ct_old -
        get_etc_value(thread_instance->etc, machine_a, task_x_current) +
        get_etc_value(thread_instance->etc, machine_a, task_y_current);

    // Maquina 2.
    machine_b_ct_old = get_machine_compute_time(selected_solution, machine_b);

    machine_b_ct_new = machine_b_ct_old -
        get_etc_value(thread_instance->etc, machine_b, task_y_current) +
        get_etc_value(thread_instance->etc, machine_b, task_x_current);

    double random;

    #ifdef CPU_MERSENNE_TWISTER
    random = cpu_mt_generate(*(thread_instance->thread_random_state));
    #endif
    #ifdef CPU_RAND
    random = cpu_rand_generate(*(thread_instance->thread_random_state));
    #endif
    #ifdef CPU_DRAND48
    random = cpu_drand48_generate(*(thread_instance->thread_random_state));
    #endif

    float swap_diff_energy;
    swap_diff_energy =
        ((machine_a_ct_old - machine_a_ct_new) * machine_a_energy_max) +
        ((machine_a_ct_new - machine_a_ct_old) * machine_a_energy_idle) +
        ((machine_b_ct_old - machine_b_ct_new) * machine_b_energy_max) +
        ((machine_b_ct_new - machine_b_ct_old) * machine_b_energy_idle);

    if ((search_type == PALS_CPU_1POP_SEARCH__MAKESPAN_GREEDY) ||
        ((random < 0.5) && (search_type == PALS_CPU_1POP_SEARCH__RANDOM_GREEDY)))
    {
        if (machine_b_ct_new <= machine_a_ct_new)
        {
            if (machine_a_ct_new < best_delta_makespan)
            {
                best_delta_makespan = machine_a_ct_new;
                best_delta_energy = swap_diff_energy;
                task_x_best_swap_pos = task_x_pos;
                task_y_best_swap_pos = task_y_pos;
                task_x_best_move_pos = -1;
                machine_b_best_move_id = -1;
            }
        }
        else if (machine_a_ct_new <= machine_b_ct_new)
        {
            if (machine_b_ct_new < best_delta_makespan)
            {
                best_delta_makespan = machine_b_ct_new;
                best_delta_energy = swap_diff_energy;
                task_x_best_swap_pos = task_x_pos;
                task_y_best_swap_pos = task_y_pos;
                task_x_best_move_pos = -1;
                machine_b_best_move_id = -1;
            }
        }
    }

    if ((search_type == PALS_CPU_1POP_SEARCH__ENERGY_GREEDY) ||
        ((random >= 0.5) && (search_type == PALS_CPU_1POP_SEARCH__RANDOM_GREEDY)))
    {
        if (swap_diff_energy > best_delta_energy)
        {
            best_delta_energy = swap_diff_energy;

            if (machine_a_ct_new <= machine_b_ct_new) best_delta_makespan = machine_b_ct_new;
            else best_delta_makespan = machine_a_ct_new;

            task_x_best_swap_pos = task_x_pos;
            task_y_best_swap_pos = task_y_pos;
            task_x_best_move_pos = -1;
            machine_b_best_move_id = -1;
        }
    }
}

void ls_best_move_simple_selection(pals_cpu_1pop_thread_arg *thread_instance, solution *selected_solution,
    int search_type, int machine_a, int machine_b_current, int task_x_pos, int task_x_current,
    float &best_delta_makespan, float &best_delta_energy, int &task_x_best_move_pos,
    int &machine_b_best_move_id, int &task_x_best_swap_pos, int &task_y_best_swap_pos)
{
    float machine_a_energy_idle = get_energy_idle_value(thread_instance->energy, machine_a);
    float machine_a_energy_max = get_energy_max_value(thread_instance->energy, machine_a);
    
    double machine_a_ct_old, machine_a_ct_new;
    double machine_b_ct_old, machine_b_ct_new;

    float current_makespan = get_makespan(selected_solution);
    //float current_energy = get_energy(selected_solution);

    double random;

    float machine_b_current_energy_idle = get_energy_idle_value(thread_instance->energy, machine_b_current);
    float machine_b_current_energy_max = get_energy_max_value(thread_instance->energy, machine_b_current);

    // Mquina 1.
    machine_a_ct_old = get_machine_compute_time(selected_solution, machine_a);
    machine_a_ct_new = machine_a_ct_old - get_etc_value(thread_instance->etc, machine_a, task_x_current);

    // Mquina 2.
    machine_b_ct_old = get_machine_compute_time(selected_solution, machine_b_current);
    machine_b_ct_new = machine_b_ct_old + get_etc_value(thread_instance->etc, machine_b_current, task_x_current);

    #ifdef CPU_MERSENNE_TWISTER
    random = cpu_mt_generate(*(thread_instance->thread_random_state));
    #endif
    #ifdef CPU_RAND
    random = cpu_rand_generate(*(thread_instance->thread_random_state));
    #endif
    #ifdef CPU_DRAND48
    random = cpu_drand48_generate(*(thread_instance->thread_random_state));
    #endif

    /*float swap_diff_energy;
    swap_diff_energy =
        ((machine_a_ct_old - machine_a_ct_new) * (machine_a_energy_max - machine_a_energy_idle)) +
        ((machine_b_ct_old - machine_b_ct_new) * (machine_b_current_energy_max - machine_b_current_energy_idle));*/

    float swap_diff_energy;
    swap_diff_energy =
        ((machine_a_ct_old - machine_a_ct_new) * machine_a_energy_max) +
        ((machine_a_ct_new - machine_a_ct_old) * machine_a_energy_idle) +
        ((machine_b_ct_old - machine_b_ct_new) * machine_b_current_energy_max) +
        ((machine_b_ct_new - machine_b_ct_old) * machine_b_current_energy_idle);

    if ((search_type == PALS_CPU_1POP_SEARCH__MAKESPAN_GREEDY) ||
        ((random < 0.5) && (search_type == PALS_CPU_1POP_SEARCH__RANDOM_GREEDY)))
    {
        if (machine_b_ct_new <= machine_a_ct_new)
        {
            if (machine_a_ct_new < best_delta_makespan)
            {
                best_delta_makespan = machine_a_ct_new;
                best_delta_energy = swap_diff_energy;
                task_x_best_swap_pos = -1;
                task_y_best_swap_pos = -1;
                task_x_best_move_pos = task_x_pos;
                machine_b_best_move_id = machine_b_current;
            }
        }
        else if (machine_a_ct_new <= machine_b_ct_new)
        {
            if (machine_b_ct_new < best_delta_makespan)
            {
                best_delta_makespan = machine_b_ct_new;
                best_delta_energy = swap_diff_energy;
                task_x_best_swap_pos = -1;
                task_y_best_swap_pos = -1;
                task_x_best_move_pos = task_x_pos;
                machine_b_best_move_id = machine_b_current;
            }
        }
    }

    if ((search_type == PALS_CPU_1POP_SEARCH__ENERGY_GREEDY) ||
        ((random >= 0.5) && (search_type == PALS_CPU_1POP_SEARCH__RANDOM_GREEDY)))
    {
        if (swap_diff_energy > best_delta_energy)
        {
            best_delta_energy = swap_diff_energy;

            if (machine_a_ct_new <= machine_b_ct_new) best_delta_makespan = machine_b_ct_new;
            else best_delta_makespan = machine_a_ct_new;

            task_x_best_swap_pos = -1;
            task_y_best_swap_pos = -1;
            task_x_best_move_pos = task_x_pos;
            machine_b_best_move_id = machine_b_current;
        }
    }
}
