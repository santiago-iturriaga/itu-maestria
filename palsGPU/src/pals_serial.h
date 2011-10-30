/*
 * pals.h
 *
 *  Created on: Jul 27, 2011
 *      Author: santiago
 */

#include "etc_matrix.h"
#include "solution.h"

#ifndef PALS_SERIAL_H_
#define PALS_SERIAL_H_

/*
 * Ejecuta PALS serial en la CPU.
 */
void pals_serial_wrapper(struct matrix *etc_matrix, struct solution *s, int &best_swaps_count, 
	int best_swaps[], float best_swaps_delta[]);

#endif /* PALS_SERIAL_H_ */
