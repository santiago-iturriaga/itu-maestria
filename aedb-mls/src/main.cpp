#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <unistd.h>
#include <mpi.h>
#include <assert.h>

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

    #ifndef NONMPI
        if (world_size < 2) {
            // Procesos insuficientes.
            fprintf(stderr, "[ERROR][%d] Debe especificar al menos 2 procesos MPI.\n", world_rank);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
    #else
        if (world_size != 1) {
            // Procesos insuficientes.
            fprintf(stderr, "[ERROR][%d] En modo NONMPI *SIEMPRE* debe especificar 1 proceso MPI.\n", world_rank);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
    #endif

    #ifndef NDEBUG
        fprintf(stderr, "[DEBUG][%d] MPI ready\n", world_rank);
    #endif

    // =============================================================
    // Loading input parameters
    // =============================================================

    #ifndef NDEBUG
        fprintf(stderr, "[DEBUG] argc = %d\n", argc);
    #endif
    
    if ((argc != 13)&&(world_rank==0)) {
        fprintf(stderr, "[ERROR] invalid arguments\n");
        fprintf(stderr, "[USAGE] %s <seed> <#iterations> <#threads> <#simulations> <density> <#reset iters> ", argv[0]);
        fprintf(stderr, "<min_coverage> <alpha> <elite> <init function> <report start s> <report every s>\n");
        fprintf(stderr, "        Init functions:\n");
        fprintf(stderr, "        0) MLS__REF_SEED\n");
        fprintf(stderr, "        1) MLS__COMPROMISE_SEED\n");
        fprintf(stderr, "        2) MLS__SUBSPACE_BASED\n");
        fprintf(stderr, "        3) MLS__RANDOM_BASED\n");
        fprintf(stderr, "        4) MLS__COMPROMISE_SEED (no nsga-ii)\n");
        fprintf(stderr, "        5) MLS__SUBSPACE_BASED (no nsga-ii)\n");
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    // Semilla aleatoria
    int seed = atoi(argv[1]);

    // Condicion de parada
    MLS.max_iterations = atoi(argv[2]);

    // Cantidad max. de hilos por procesos MPI
    MLS.count_threads = atoi(argv[3]);

    if (MLS.count_threads > MLS__MAX_THREADS) {
        fprintf(stderr, "[ERROR][%d] La cantidad máxima de hilos en cada proceso MPI es de %d.\n", world_rank, MLS__MAX_THREADS);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    // Config. NS3
    // Number of independent runs of the simulator, if > 1, the results given are averaged over all the runs
    // MLS.simul_runs = 10;
    MLS.simul_runs = atoi(argv[4]);
    // 25 for 100 devices/km^2 density, 50 for 200, and 75 for 300
    // MLS.number_devices = 25;
    MLS.number_devices = atoi(argv[5]);
    
    MLS.count_reset = atoi(argv[6]);
    MLS.min_coverage = atoi(argv[7]);
    MLS.alpha = atof(argv[8]);
    
    MLS.elite = (atoi(argv[9]) != 0);
    MLS.init_func = atoi(argv[10]);
    
    assert(MLS.init_func >= 0);
    assert(MLS.init_func <= 5);
    
    AGA.report_start = atof(argv[11]);
    AGA.report_every = atof(argv[12]);
    
    if (MLS.simul_runs % 2 != 0) {
        fprintf(stderr, "[ERROR][%d] La cantidad de simulaciones debe ser par.\n", world_rank);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    // Dominio de las variables de búsqueda.
    MLS.lbound_min_delay = 0;
    MLS.ubound_min_delay = 1;
    MLS.lbound_max_delay = 0;
    MLS.ubound_max_delay = 5;
    MLS.lbound_border_threshold = -95;
    MLS.ubound_border_threshold = -70;
    MLS.lbound_margin_threshold = 0;
    MLS.ubound_margin_threshold = 3;
    MLS.lbound_neighbors_threshold = 0;
    MLS.ubound_neighbors_threshold = 50;

    if (world_rank == 0) {
        fprintf(stderr, "===========================================================\n");
        fprintf(stderr, "Configuración\n");
        fprintf(stderr, "===========================================================\n");
        fprintf(stderr, "   Random seed        = %d\n", seed);
        fprintf(stderr, "   MPI size           = %d\n", world_size);
        fprintf(stderr, "   MLS.count_threads  = %d\n", MLS.count_threads);
        fprintf(stderr, "   MLS.max_iterations = %d\n", MLS.max_iterations);
        fprintf(stderr, "   MLS.number_devices = %d\n", MLS.number_devices);
        fprintf(stderr, "   MLS.simul_runs     = %d\n", MLS.simul_runs);
        
        fprintf(stderr, "   MLS.count_reset    = %d\n", MLS.count_reset);
        fprintf(stderr, "   MLS.min_coverage   = %d\n", MLS.min_coverage);
        fprintf(stderr, "   MLS.alpha          = %f\n", MLS.alpha);
        fprintf(stderr, "   MLS.elite          = %d\n", MLS.elite);
        
        if (MLS.init_func == 0) {
            fprintf(stderr, "   MLS.init_func      = MLS__REF_SEED\n");
        } else if (MLS.init_func == 1) {
            fprintf(stderr, "   MLS.init_func      = MLS__COMPROMISE_SEED\n");
        } else if (MLS.init_func == 2) {
            fprintf(stderr, "   MLS.init_func      = MLS__SUBSPACE_BASED\n");
        } else if (MLS.init_func == 3) {
            fprintf(stderr, "   MLS.init_func      = MLS__RANDOM\n");            
        } else if (MLS.init_func == 4) {
            fprintf(stderr, "   MLS.init_func      = MLS__COMPROMISE_SEED (no nsga-ii)\n");
        } else if (MLS.init_func == 5) {
            fprintf(stderr, "   MLS.init_func      = MLS__SUBSPACE_BASED (no nsga-ii)\n");
        }
        
        fprintf(stderr, "   AGA.report_start   = %f\n", AGA.report_start);
        fprintf(stderr, "   AGA.report_every   = %f\n", AGA.report_every);
        
        fprintf(stderr, "===========================================================\n");
    }

    // =============================================================
    // Inicializo MPI
    // =============================================================

    // Inicializo el tipo de datos SOLUTION en MPI.
    init_mpi_solution(MLS.count_threads);

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

    #ifndef NONMPI
        if (world_rank == 0) {
            // =============================================================
            // Proceso AGA
            // =============================================================

            archivers_aga(seed);

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
            mls(seed + world_rank, &mls_comm);

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
    #else
        // =============================================================
        // NONMPI: Solving the problem.
        // =============================================================
        mls(seed + world_rank, &mls_comm);
    #endif

    #ifdef MPI_MODE_BUFFERED
        MPI_Buffer_detach(mpi_recv_buffer, &mpi_buffer_size);
        free(mpi_recv_buffer);
    #endif

    MPI_Finalize();

    return EXIT_SUCCESS;
}
