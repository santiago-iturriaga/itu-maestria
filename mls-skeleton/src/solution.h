/*
 * Handles a solution in memory.
 */

#include <stdio.h>

#ifndef SOLUTION_H_
#define SOLUTION_H_

#define OBJECTIVES 2
#define SOLUTION__MAKESPAN_OBJ  0
#define SOLUTION__ENERGY_OBJ    1

#define SOLUTION__STATUS_TO_DEL     -2
#define SOLUTION__STATUS_NOT_READY  -1
#define SOLUTION__STATUS_EMPTY      0
#define SOLUTION__STATUS_NEW        1
#define SOLUTION__STATUS_READY      2

struct solution {
    int status;
    
    /* Estructura que representa una solución */
};

/* Crea una copia exacta de la solución "src" en la solución "dst" */
void clone_solution(struct solution *dst, struct solution *src);

/* Devuelve el valor del la n-esima métrica objetivo en la solución s */
float get_objective(struct solution *s, int obj_index);

#endif
