#include "../pals_cpu_1pop.h"

#ifndef MACHINE_SEL_SIMPLE__H
#define MACHINE_SEL_SIMPLE__H

void machines_selection(pals_cpu_1pop_thread_arg *thread_instance, solution *selected_solution,
    int search_type, int &machine_a, int &machine_b);

#endif // MACHINE_SEL_SIMPLE__H
