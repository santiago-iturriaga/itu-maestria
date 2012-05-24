/*
 * MCT heuristic basic implementation.
 */

#include "../etc_matrix.h"
#include "../solution.h"

#ifndef MCT_H_
#define MCT_H_

void compute_mct(struct solution *solution);
void compute_custom_mct(struct solution *solution, int starting_task);

#endif
