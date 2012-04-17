#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <limits.h>

#include "billionga.h"

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stdout, "Wrong! RFM!\n Usage: %s <problem size>\n\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    long int problem_size;
    problem_size = atol(argv[1]);
    
    return EXIT_SUCCESS;
}
