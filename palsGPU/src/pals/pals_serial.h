/*
 * pals.h
 *
 *  Created on: Jul 27, 2011
 *      Author: santiago
 */

#include "../etc_matrix.h"
#include "../solution.h"

#ifndef PALS_SERIAL_H_
#define PALS_SERIAL_H_

void pals_serial_wrapper(struct matrix *etc_matrix, struct solution *s, int &best_swap_task_a, int &best_swap_task_b, float &best_swap_delta);

/*
 * Ejecuta el algoritmo.
 * Búsqueda serial sobre el todo el dominio del problema.
 */
void pals_serial(struct params &input, struct matrix *etc_matrix, struct solution *current_solution);

#endif /* PALS_SERIAL_H_ */
