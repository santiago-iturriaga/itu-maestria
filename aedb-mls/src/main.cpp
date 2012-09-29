#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <unistd.h>
#include <mpi.h>

#include "solution.h"
#include "config.h"

#include "mls/mls.h"
#include "aga/aga.h"

int main(int argc, char** argv)
{
    int world_rank, world_size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    if (world_size < 2) {
        // Procesos insuficientes.
        fprintf(stderr, "[ERROR] Debe especificar al menos 2 procesos MPI.\n");
        exit(EXIT_FAILURE);
    }
    
    // =============================================================
    // Loading input parameters
    // =============================================================
    // ...

    if (world_rank == 0) {
        // =============================================================
        // Proceso AGA
        // =============================================================
        
    } else {
        // =============================================================
        // Proceso MLS
        // =============================================================
        
        // =============================================================
        // Loading problem instance
        // =============================================================
        // ...
        
        // =============================================================
        // Solving the problem.
        // =============================================================
        mls();
        
        // =============================================================
        // Release memory
        // =============================================================
        // ...
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
