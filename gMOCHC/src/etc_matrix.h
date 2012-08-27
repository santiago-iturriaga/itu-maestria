#include "config.h"
#include "load_params.h"

#ifndef ETC_MATRIX_H_
#define ETC_MATRIX_H_

struct etc_matrix {
    int tasks_count;
    int machines_count;
    
    FLOAT* data;
    int* data_machine_index;
};

void init_etc_matrix(struct params *input, struct etc_matrix *etc);
void free_etc_matrix(struct etc_matrix *etc);
void show_etc_matrix(struct etc_matrix *etc);

void set_etc_value(struct etc_matrix *etc, int machine, int task, FLOAT value);
FLOAT get_etc_value(struct etc_matrix *etc, int machine, int task);

#endif /* ETC_MATRIX_H_ */
