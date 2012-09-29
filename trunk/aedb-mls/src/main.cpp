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

int world_rank, world_size, err;
char machine_name[180];

int main(int argc, char** argv)
{
    err = MPI_Init(&argc, &argv);
    if (err != MPI_SUCCESS) {
        fprintf(stderr, "[ERROR] MPI_Init failed!\n");
        exit(EXIT_FAILURE);
    }
  
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (err != MPI_SUCCESS) {
        fprintf(stderr, "[ERROR] MPI_Comm_size failed!\n");
        exit(EXIT_FAILURE);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (err != MPI_SUCCESS) {
        fprintf(stderr, "[ERROR] MPI_Comm_rank failed!\n");
        exit(EXIT_FAILURE);
    }
    
    int aux;
    MPI_Get_processor_name(machine_name, &aux);
    
    if (world_size < 2) {
        // Procesos insuficientes.
        fprintf(stderr, "[ERROR][%d] Debe especificar al menos 2 procesos MPI.\n", world_rank);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
 
    #ifndef NDEBUG
        fprintf(stderr, "[DEBUG][%d] MPI ready\n", world_rank);
    #endif
       
    // =============================================================
    // Loading input parameters
    // =============================================================
    // Cantidad max. de hilos por procesos MPI
    MLS.count_threads = 2;
    
    // Condicion de parada
    MLS.max_iterations = 100;
    
    if (MLS.count_threads > MLS__MAX_THREADS) {
        fprintf(stderr, "[ERROR][%d] La cantidad máxima de hilos en cada proceso MPI es de %d.\n", world_rank, MLS__MAX_THREADS);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    // =============================================================
    // Inicializo MPI
    // =============================================================
    
    // Inicializo el tipo de datos SOLUTION en MPI.
    init_mpi_solution();
    
    // Construyo el grupo MLS
    MPI_Group world_group, mls_group;
    MPI_Comm mls_comm;
    
    int aga_rank[1] = {0};
    
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group_excl(world_group, 1, aga_rank, &mls_group);
    MPI_Comm_create(MPI_COMM_WORLD, mls_group, &mls_comm);
    
    if (world_rank == 0) {
        // =============================================================
        // Proceso AGA
        // =============================================================
        
        archivers_aga();
        
        // Finalización...        
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
        mls(0);
        
        // Espero a que todos los MLS terminen.
        MPI_Barrier(mls_comm);
        
        // Le aviso a AGA que puede terminar.
        if (world_rank == 1) {
            #ifndef NDEBUG
                fprintf(stderr, "[DEBUG][%d] Sending terminate message to process AGA\n", world_rank);
            #endif
            MPI_Send(aga_rank, 1, MPI_INT, AGA__PROCESS_RANK, AGA__EXIT_MSG, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
