/*
 * Parallel MINMIN heuristic implementation.
 */

#include "../etc_matrix.h"
#include "../solution.h"

#ifndef PMINMIN_H_
#define PMINMIN_H_

struct threadData {
    int t_i;
    int t_f;
    struct matrix *etc;
    struct solution *sol;
};

void compute_pminmin(struct matrix *etc_matrix, struct solution *sol, int numberOfThreads);
void * compute_pminmin_thread(void *);

#endif
