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
    MLS.max_iterations = 10000;
    
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
    
    #ifdef MPI_MODE_BUFFERED
        int mpi_buffer_size;
        char *mpi_recv_buffer;
    
        int solutions_in_buffer = MLS__BUFFER_SIZE * MLS.count_threads;
        
        MPI_Pack_size(solutions_in_buffer, mpi_solution_type, MPI_COMM_WORLD, &mpi_buffer_size);
        mpi_buffer_size += MLS__BUFFER_SIZE * MPI_BSEND_OVERHEAD;
        mpi_recv_buffer = (char*)malloc(mpi_buffer_size);
        
        #ifndef NDEBUG
            fprintf(stderr, "[INFO][%d] MPI buffer = %d soluciones (%d bytes)\n", world_rank, solutions_in_buffer, mpi_buffer_size);
        #endif
        
        MPI_Buffer_attach(mpi_recv_buffer, mpi_buffer_size);
    #endif
    
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
        int seed = 0;
        
        mls(seed + world_rank);
        
        // Espero a que todos los MLS terminen.
        MPI_Barrier(mls_comm);
        
        // Le aviso a AGA que puede terminar.
        if (world_rank == 1) {
            #ifndef NDEBUG
                fprintf(stderr, "[DEBUG][%d] Sending terminate message to process AGA\n", world_rank);
            #endif
                        
            #ifdef MPI_MODE_STANDARD
                MPI_Send(aga_rank, 1, MPI_INT, AGA__PROCESS_RANK, AGA__EXIT_MSG, MPI_COMM_WORLD);
            #endif
            
            #ifdef MPI_MODE_SYNC
                MPI_Ssend(aga_rank, 1, MPI_INT, AGA__PROCESS_RANK, AGA__EXIT_MSG, MPI_COMM_WORLD);
            #endif
            
            #ifdef MPI_MODE_BUFFERED
                MPI_Bsend(aga_rank, 1, MPI_INT, AGA__PROCESS_RANK, AGA__EXIT_MSG, MPI_COMM_WORLD);
            #endif
        }
    }
   
    #ifdef MPI_MODE_BUFFERED
        MPI_Buffer_detach(mpi_recv_buffer, &mpi_buffer_size);
        free(mpi_recv_buffer);
    #endif
    
    MPI_Finalize();

    return EXIT_SUCCESS;
}
