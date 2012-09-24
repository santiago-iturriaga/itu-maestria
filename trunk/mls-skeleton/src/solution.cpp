#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "config.h"
#include "solution.h"

void clone_solution(struct solution *dst, struct solution *src) {

}

float get_objective(struct solution *s, int obj_index) {
    if (obj_index == SOLUTION__MAKESPAN_OBJ) {
        return 0; //s->__makespan;
    } else if (obj_index == SOLUTION__ENERGY_OBJ) {
        return 0; //s->__total_energy_consumption;
    } else {
        assert(false);
    }
}
