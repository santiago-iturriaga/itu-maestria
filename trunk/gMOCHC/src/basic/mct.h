/*
 * MCT heuristic basic implementation.
 */

#include "../solution.h"

#ifndef MCT_H_
#define MCT_H_

void compute_mct(struct solution *sol);
void compute_mct_random(struct solution *sol, int start, int direction);

#endif
