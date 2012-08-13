#if !defined(CMOCHC_ISLANDS__H)
#define CMOCHC_ISLANDS__H

#include "../load_params.h"
#include "../scenario.h"
#include "../etc_matrix.h"
#include "../energy_matrix.h"

void compute_cmochc_island(struct params &input, struct scenario &current_scenario, 
    struct etc_matrix &etc, struct energy_matrix &energy);

#endif // CMOCHC_ISLANDS__H
