#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <cuda.h>

#include "config.h"
#include "billionga.h"

inline int termination_criteria_met(struct bga_state *problem_state) {
    // ...
    return true;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stdout, "Wrong! RFM!\n\nUsage: %s <problem size>\n(where: 1 <= problem size <= %ld)\n\n", argv[0], LONG_MAX);
        return EXIT_FAILURE;
    }
    
    long int problem_size;
    problem_size = atol(argv[1]);
    
    #ifdef INFO
    fprintf(stdout, "[INFO] Problem size: %ld\n", problem_size);
    #endif
    
    struct bga_state problem_state;
    bga_initialization(&problem_state);
    
    // ...
    fprintf(stdout, "%ld %ld\n", sizeof(float), sizeof(double));
    
    bga_free(&problem_state);
    
    return EXIT_SUCCESS;
}
