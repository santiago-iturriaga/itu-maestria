#if !defined(CMOCHC_CELL__H)
#define CMOCHC_CELL__H

#include "../load_params.h"
#include "../scenario.h"
#include "../etc_matrix.h"
#include "../energy_matrix.h"

void compute_cmochc_cell(struct params &input, struct scenario &current_scenario, 
    struct etc_matrix &etc, struct energy_matrix &energy);

#endif // CMOCHC_CELL__H
