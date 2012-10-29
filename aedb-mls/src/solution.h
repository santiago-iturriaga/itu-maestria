/*
 * Handles a solution in memory.
 */

#include <stdio.h>
#include <assert.h>

#ifndef SOLUTION_H_
#define SOLUTION_H_

#define OBJECTIVES 3
#define SOLUTION__ENERGY_OBJ        0
#define SOLUTION__COVERAGE_OBJ      1
#define SOLUTION__NFORWARDINGS_OBJ  2

#define SOLUTION__STATUS_TO_DEL     -2
#define SOLUTION__STATUS_NOT_READY  -1
#define SOLUTION__STATUS_EMPTY      0
#define SOLUTION__STATUS_NEW        1
#define SOLUTION__STATUS_READY      2

/* Estructura que representa una solución */
struct solution {
    int status;
    
    /* Solución */
    double borders_threshold;
    double margin_forwarding;
    double min_delay;
    double max_delay;
    int neighbors_threshold;

    /* Evaluación de la solución */
    double energy;
    double coverage;
    double nforwardings;
    double time;
};

extern MPI_Datatype mpi_solution_type;
extern MPI_Datatype mpi_solution_type_array;

void init_mpi_solution(int array_size);

/* Crea una copia exacta de la solución "src" en la solución "dst" */
void clone_solution(struct solution *dst, struct solution *src);

void show_solution(struct solution *sol);

/* Devuelve el valor del la n-esima métrica objetivo en la solución s */
inline double get_objective(struct solution *s, int obj_index) {
    if (obj_index == SOLUTION__ENERGY_OBJ) {
        return s->energy;
    } else if (obj_index == SOLUTION__COVERAGE_OBJ) {
        return (double)(s->coverage);
    } else if (obj_index == SOLUTION__NFORWARDINGS_OBJ) {
        return (double)(s->nforwardings);
    } else {
        assert(false);
        return 0;
    }
}

#endif
