#include "../pals_cpu_1pop.h"

#ifndef EVOL_GUIDE_COMPLEX__H
#define EVOL_GUIDE_COMPLEX__H

void ls_best_swap_selection(pals_cpu_1pop_thread_arg *thread_instance, solution *selected_solution,
    int search_type, int machine_a, int machine_b, int task_x_pos, int task_x_current,
    int task_y_pos, int task_y_current, float &best_delta_makespan, float &best_delta_energy,
    int &task_x_best_move_pos, int &machine_b_best_move_id, int &task_x_best_swap_pos, int &task_y_best_swap_pos);

void ls_best_move_selection(pals_cpu_1pop_thread_arg *thread_instance, solution *selected_solution,
    int search_type, int machine_a, int machine_b_current, int task_x_pos, int task_x_current,
    float &best_delta_makespan, float &best_delta_energy, int &task_x_best_move_pos,
    int &machine_b_best_move_id, int &task_x_best_swap_pos, int &task_y_best_swap_pos);

#endif // EVOL_GUIDE_COMPLEX__H
