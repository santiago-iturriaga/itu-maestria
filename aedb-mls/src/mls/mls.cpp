#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <assert.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <mpi.h>

#include "mls.h"

#include "../config.h"
#include "../random/cpu_mt.h"

#ifdef NONMPI
    #include "../aga/aga.h"
#endif

// Coverage 17%
double INIT_REF_MIN_DELAY = 0;
double INIT_REF_MAX_DELAY = 1;
double INIT_REF_BORDERS = -90;
double INIT_REF_MARGIN = 0.5;
double INIT_REF_NEIGH = 12;

// Coverage 22%
//double INIT_REF_MIN_DELAY = 0;
//double INIT_REF_MAX_DELAY = 0.5;
//double INIT_REF_BORDERS = -85;
//double INIT_REF_MARGIN = 0.5;
//double INIT_REF_NEIGH = 12;

// NSGA-II

//#define INIT_SOLS_COUNT 6
// min_delay	max_delay	borders_threshold	margin_forwarding	neighbors_threshold
// 100
//0.5984374114	0.2178867285	-72.8205269102	2.3018358858	0.4019944967
//0.1153369107	1.3178490859	-93.468117255	1.7626870955	48.283330341
//0.1156424482	0.3551939489	-87.4881408056	2.2399798063	17.3400943663
// 200
//0.1557322416	0.056797501	-94.90843399	0.1629409934	36.4882745701
//0.0329722337	0.8244608171	-94.544613148	0.0931650812	21.7951438837
//0.0038013151	0.1541347337	-92.9160964977	0.6795834844	9.3808928816
//double INIT_MIN_DELAY[6] = {0.5984374114,0.1153369107,0.1156424482,0.1557322416,0.0329722337,0.0038013151};
//double INIT_MAX_DELAY[6] = {0.2178867285,1.3178490859,0.3551939489,0.056797501,0.8244608171,0.1541347337};
//double INIT_BORDERS[6] = {-72.8205269102,-93.468117255,-87.4881408056,-94.90843399,-94.544613148,-92.9160964977};
//double INIT_MARGIN[6] = {2.3018358858,1.7626870955,2.2399798063,0.1629409934,0.0931650812,0.6795834844};
//double INIT_NEIGH[6] = {0.4019944967,48.283330341,17.3400943663,36.4882745701,21.7951438837,9.3808928816};

#define INIT_SOLS_COUNT 9
// min_delay	max_delay	borders_threshold	margin_forwarding	neighbors_threshold
// 100
//0.1890385049	0.2874826665	-87.9702123612	0.1761382193	33.4085695899
//0.1939969439	0.5229135688	-92.5437063962	0.1992473382	25.8326657156
//0.3648123217	1.0725528937	-94.1689780395	0.0588310505	28.2189752797
// 200
//0.0234114659	0.5515028066	-94.5265388391	0.0657448452	11.802192724
//0.0610926247	0.2103036896	-94.8574254726	0.1573215758	36.4714263463
//0.0512337975	0.0326064991	-94.9918974897	0.0737254019	3.9729921177
// 300
//0.077061695	1.534922946	-94.7633024778	0.0249335585	12.8804501704
//0.0134215953	0.360309296	-70.5085186529	0.7202181484	14.0681387339
//0.2590408908	0.3988473201	-71.4049834319	0.193549638	13.7669209882

//nuevo ns25
//0.1894764949	0.2960242698	-87.9760410056	0.0946339108	28.3942710238

double INIT_MIN_DELAY[9] = {
    //0.1890385049,0.1939969439,0.3648123217,
    0.364812,0.072915,0.168417,
    0.271577961786971,0.0804609733661016,0.388529079531146,
    0.077061695,0.111884612483053,0.163459702145845
    };
double INIT_MAX_DELAY[9] = {
    //0.2874826665,0.5229135688,1.0725528937,
    1.072553,0.112087,0.21783,
    0.550888508532971,0.298916933956612,0.450202296454503,
    1.534922946,1.64279398343566,0.71657297798839
    };
double INIT_BORDERS[9] = {
    //-87.9702123612,-92.5437063962,-94.1689780395,
    -94.168978,-91.816182,-88.092821,
    -93.117789801688,-94.1090433460745,-94.6979965646302,
    -94.7633024778,-94.7011702976319,-94.0069142089263
    };
double INIT_MARGIN[9] = {
    //0.1761382193,0.1992473382,0.0588310505,
    0.058831,0.720218,0.720218,
    0.176047953330046,0.696763328269848,0.245657644799441,
    0.0249335585,0.0856863132957332,0.726057856272232
    };
double INIT_NEIGH[9] = {
    //33.4085695899,25.8326657156,28.2189752797,
    28,2,7,
    6.20200316040405,5.97635814105392,9.98416357531338,
    12.8804501704,21.718301638662,12.9106523155023
    };

struct mls_instance MLS;

/*
 * Reserva e inicializa la memoria con los datos del problema.
 */
void mls_init(int seed);

/*
 * Libera la memoria pedida durante la inicialización.
 */
void mls_finalize();

/*
 * Ejecuta un hilo de la búsqueda.
 */
void* mls_thread(void *data);

void mls(int seed, MPI_Comm *mls_comm)
{
    // Inicializo la memoria y los hilos de ejecucion.
    MLS.mls_comm = mls_comm;
    mls_init(seed);

    // Mientras los hilos buscan no tengo nada mas que hacer.
    // Espero a que terminen....

    // Bloqueo la ejecucion hasta que terminen todos los hilos.
    for(int i = 0; i < MLS.count_threads; i++)
    {
        if(pthread_join(MLS.threads[i], NULL))
        {
            fprintf(stderr, "[ERROR][%d] Could not join thread %d\n", world_rank, i);
            exit(EXIT_FAILURE);
        }
    }

    int total_iterations = 0;

    for (int i = 0; i < MLS.count_threads; i++)
    {
        total_iterations += MLS.total_iterations[i];
    }

    fprintf(stdout, "[INFO][%d] Cantidad de iteraciones: %d\n", world_rank, total_iterations);

    // Libero la memoria pedida en la inicialización.
    mls_finalize();
}

void mls_init(int seed)
{
    // =========================================================================
    // Inicializo las semillas para la generación de numeros aleatorios.
    srand(seed);
    long int random_seed;

    for (int i = 0; i < MLS.count_threads; i++)
    {
        random_seed = rand();
        cpu_mt_init(random_seed, MLS.random_states[i]);
    }

    // =========================================================================
    // Creo e inicializo los threads y los mecanismos de sincronizacion.
    for (int i = 0; i < MLS.count_threads; i++) {
        if (pthread_mutex_init(&(MLS.work_type_mutex[i]), NULL))
        {
            fprintf(stderr, "[ERROR][%d] Could not create work type mutex\n", world_rank);
            exit(EXIT_FAILURE);
        }
    }

    if (pthread_mutex_init(&(MLS.mpi_mutex), NULL))
    {
        fprintf(stderr, "[ERROR][%d] Could not create mpi mutex\n", world_rank);
        exit(EXIT_FAILURE);
    }

    if (pthread_barrier_init(&(MLS.sync_barrier), NULL, MLS.count_threads))
    {
        fprintf(stderr, "[ERROR][%d] Could not create sync barrier\n", world_rank);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < MLS.count_threads; i++)
    {
        MLS.threads_id[i] = i;
        MLS.work_type[i] = MLS__INIT;

        if (pthread_create(&MLS.threads[i], NULL, mls_thread,  (void*) &MLS.threads_id[i]))
        {
            fprintf(stderr, "[ERROR][%d] Could not create slave thread %d\n", world_rank, i);
            exit(EXIT_FAILURE);
        }
    }
}

void mls_finalize()
{
    for (int i = 0; i < MLS.count_threads; i++) {
        pthread_mutex_destroy(&(MLS.work_type_mutex[i]));
    }

    pthread_mutex_destroy(&(MLS.mpi_mutex));
    pthread_barrier_destroy(&(MLS.sync_barrier));
}

void* mls_thread(void *data)
{
    int thread_id = *((int*)data);

    #ifndef NDEBUG
        fprintf(stderr, "[DEBUG][%d] Thread id: %d\n", world_rank, thread_id);
    #endif

    MLS.total_iterations[thread_id] = 0;

    MPI_Status status;

    int terminate = 0;
    int work_iteration_size;
    int search_iteration;
    double random = 0;
    int work_type;

    char ns3_line[256];
    char ns3_command[1024];

    int reset_hit = MLS.count_reset;

    pthread_mutex_lock(&MLS.work_type_mutex[thread_id]);
        work_type = MLS.work_type[thread_id];
    pthread_mutex_unlock(&MLS.work_type_mutex[thread_id]);

    int half_simul_runs;
    half_simul_runs = MLS.simul_runs / 2;

    while ((terminate == 0) && (MLS.total_iterations[thread_id] < MLS.max_iterations))
    {
        #ifndef NDEBUG
            if (thread_id == 0) {
                fprintf(stderr, "[DEBUG] (%d)\n", MLS.total_iterations[thread_id]);
            }
        #endif

        if (work_type == MLS__EXIT)
        {
            // =================================================================
            // Finalizo la ejecucion del hilo!
            terminate = 1;
        }
        else if (work_type == MLS__INIT)
        {
            #ifndef NDEBUG
                fprintf(stderr, "[DEBUG] MLS__INIT\n");
            #endif

            // =================================================================
            // Inicializo un individuo con una heurística.

            if (MLS.init_func == 0) { // MLS__REF_SEED
                if (world_rank == 0 && thread_id == 0) {
                    MLS.population[thread_id].min_delay = INIT_REF_MIN_DELAY;
                    MLS.population[thread_id].max_delay = INIT_REF_MAX_DELAY;
                    MLS.population[thread_id].borders_threshold = INIT_REF_BORDERS;
                    MLS.population[thread_id].margin_forwarding = INIT_REF_MARGIN;
                    MLS.population[thread_id].neighbors_threshold = INIT_REF_NEIGH;
                } else {
                    MLS.population[thread_id].min_delay = INIT_REF_MIN_DELAY *
                        (1.1 - (cpu_mt_generate(MLS.random_states[thread_id]) * 0.2)); /* value * [0.9,1.1) */

                    MLS.population[thread_id].max_delay = INIT_REF_MAX_DELAY *
                        (1.1 - (cpu_mt_generate(MLS.random_states[thread_id]) * 0.2));

                    MLS.population[thread_id].borders_threshold = INIT_REF_BORDERS *
                        (1.1 - (cpu_mt_generate(MLS.random_states[thread_id]) * 0.2));

                    MLS.population[thread_id].margin_forwarding = INIT_REF_MARGIN *
                        (1.1 - (cpu_mt_generate(MLS.random_states[thread_id]) * 0.2));

                    MLS.population[thread_id].neighbors_threshold = INIT_REF_NEIGH *
                        (1.1 - (cpu_mt_generate(MLS.random_states[thread_id]) * 0.2));
                }
            } else if ((MLS.init_func == 1)||(MLS.init_func == 4)) { // MLS__COMPROMISE_SEED
                if ((world_rank < INIT_SOLS_COUNT) && (thread_id == 0) && (MLS.init_func == 1)) {
                    MLS.population[thread_id].min_delay = INIT_MIN_DELAY[world_rank];
                    MLS.population[thread_id].max_delay = INIT_MAX_DELAY[world_rank];
                    MLS.population[thread_id].borders_threshold = INIT_BORDERS[world_rank];
                    MLS.population[thread_id].margin_forwarding = INIT_MARGIN[world_rank];
                    MLS.population[thread_id].neighbors_threshold = INIT_NEIGH[world_rank];
                } else {
                    MLS.population[thread_id].min_delay = INIT_REF_MIN_DELAY *
                        (1.1 - (cpu_mt_generate(MLS.random_states[thread_id]) * 0.2)); /* value * [0.9,1.1) */

                    MLS.population[thread_id].max_delay = INIT_REF_MAX_DELAY *
                        (1.1 - (cpu_mt_generate(MLS.random_states[thread_id]) * 0.2));

                    MLS.population[thread_id].borders_threshold = INIT_REF_BORDERS *
                        (1.1 - (cpu_mt_generate(MLS.random_states[thread_id]) * 0.2));

                    MLS.population[thread_id].margin_forwarding = INIT_REF_MARGIN *
                        (1.1 - (cpu_mt_generate(MLS.random_states[thread_id]) * 0.2));

                    MLS.population[thread_id].neighbors_threshold = INIT_REF_NEIGH *
                        (1.1 - (cpu_mt_generate(MLS.random_states[thread_id]) * 0.2));
                }
            } else if ((MLS.init_func == 2)||(MLS.init_func == 5)) { // MLS__SUBSPACE_BASED
                #ifndef NDEBUG
                    fprintf(stderr, "[DEBUG] MLS__SUBSPACE_BASED\n");
                #endif
                
                if ((world_rank < INIT_SOLS_COUNT) && (thread_id == 0) && (MLS.init_func == 2)) {                   
                    MLS.population[thread_id].min_delay = INIT_MIN_DELAY[world_rank];
                    MLS.population[thread_id].max_delay = INIT_MAX_DELAY[world_rank];
                    MLS.population[thread_id].borders_threshold = INIT_BORDERS[world_rank];
                    MLS.population[thread_id].margin_forwarding = INIT_MARGIN[world_rank];
                    MLS.population[thread_id].neighbors_threshold = INIT_NEIGH[world_rank];
                } else {
                    int pop_index = thread_id + (world_rank * MLS.count_threads);

                    int selected;
                    selected = (int)(cpu_mt_generate(MLS.random_states[thread_id]) * INIT_SOLS_COUNT);

                    MLS.population[thread_id].min_delay = INIT_MIN_DELAY[selected] +
                        ((pop_index + cpu_mt_generate(MLS.random_states[thread_id]))/((world_rank + 1) *
                        MLS.count_threads)) * (MLS.ubound_min_delay - MLS.lbound_min_delay);

                    MLS.population[thread_id].max_delay = INIT_MAX_DELAY[selected] *
                        ((pop_index + cpu_mt_generate(MLS.random_states[thread_id]))/((world_rank + 1) *
                        MLS.count_threads)) * (MLS.ubound_max_delay - MLS.lbound_max_delay);

                    MLS.population[thread_id].borders_threshold = INIT_BORDERS[selected] *
                        ((pop_index + cpu_mt_generate(MLS.random_states[thread_id]))/((world_rank + 1) *
                        MLS.count_threads)) * (MLS.ubound_border_threshold - MLS.lbound_border_threshold);

                    MLS.population[thread_id].margin_forwarding = INIT_MARGIN[selected] *
                        ((pop_index + cpu_mt_generate(MLS.random_states[thread_id]))/((world_rank + 1) *
                        MLS.count_threads)) * (MLS.ubound_margin_threshold - MLS.lbound_margin_threshold);

                    MLS.population[thread_id].neighbors_threshold = INIT_NEIGH[selected] *
                        ((pop_index + cpu_mt_generate(MLS.random_states[thread_id]))/((world_rank + 1) *
                        MLS.count_threads)) * (MLS.ubound_neighbors_threshold - MLS.lbound_neighbors_threshold);
                }
            } else if (MLS.init_func == 3) { // MLS__RANDOM_BASED
                if ((world_rank < INIT_SOLS_COUNT) && (thread_id == 0)) {
                    MLS.population[thread_id].min_delay = INIT_MIN_DELAY[world_rank];
                    MLS.population[thread_id].max_delay = INIT_MAX_DELAY[world_rank];
                    MLS.population[thread_id].borders_threshold = INIT_BORDERS[world_rank];
                    MLS.population[thread_id].margin_forwarding = INIT_MARGIN[world_rank];
                    MLS.population[thread_id].neighbors_threshold = INIT_NEIGH[world_rank];
                } else {
                    MLS.population[thread_id].min_delay = MLS.lbound_min_delay +
                        cpu_mt_generate(MLS.random_states[thread_id]) * (MLS.ubound_min_delay - MLS.lbound_min_delay);
                    MLS.population[thread_id].max_delay = MLS.lbound_max_delay +
                        cpu_mt_generate(MLS.random_states[thread_id]) * (MLS.ubound_max_delay - MLS.lbound_max_delay);
                    MLS.population[thread_id].borders_threshold = MLS.lbound_border_threshold +
                        cpu_mt_generate(MLS.random_states[thread_id]) * (MLS.ubound_border_threshold - MLS.lbound_border_threshold);
                    MLS.population[thread_id].margin_forwarding = MLS.lbound_margin_threshold +
                        cpu_mt_generate(MLS.random_states[thread_id]) * (MLS.ubound_margin_threshold - MLS.lbound_margin_threshold);
                    MLS.population[thread_id].neighbors_threshold = MLS.lbound_neighbors_threshold +
                        cpu_mt_generate(MLS.random_states[thread_id]) * (MLS.ubound_neighbors_threshold - MLS.lbound_neighbors_threshold);
                }
            }

            if (MLS.population[thread_id].min_delay > MLS.ubound_min_delay) {
                MLS.population[thread_id].min_delay = MLS.ubound_min_delay;
            } else if (MLS.population[thread_id].min_delay < MLS.lbound_min_delay) {
                MLS.population[thread_id].min_delay = MLS.lbound_min_delay;
            }

            if (MLS.population[thread_id].max_delay > MLS.ubound_max_delay) {
                MLS.population[thread_id].max_delay = MLS.lbound_max_delay;
            } else if (MLS.population[thread_id].max_delay < MLS.lbound_max_delay) {
                MLS.population[thread_id].max_delay = MLS.lbound_max_delay;
            }

            if (MLS.population[thread_id].borders_threshold > MLS.ubound_border_threshold) {
                MLS.population[thread_id].borders_threshold = MLS.ubound_border_threshold;
            } else if (MLS.population[thread_id].borders_threshold < MLS.lbound_border_threshold) {
                MLS.population[thread_id].borders_threshold = MLS.lbound_border_threshold;
            }

            if (MLS.population[thread_id].margin_forwarding > MLS.ubound_margin_threshold) {
                MLS.population[thread_id].margin_forwarding = MLS.ubound_margin_threshold;
            } else if (MLS.population[thread_id].margin_forwarding < MLS.lbound_margin_threshold) {
                MLS.population[thread_id].margin_forwarding = MLS.lbound_margin_threshold;
            }

            if (MLS.population[thread_id].neighbors_threshold > MLS.ubound_neighbors_threshold) {
                MLS.population[thread_id].neighbors_threshold = MLS.ubound_neighbors_threshold;
            } else if (MLS.population[thread_id].neighbors_threshold < MLS.lbound_neighbors_threshold) {
                MLS.population[thread_id].neighbors_threshold = MLS.lbound_neighbors_threshold;
            }

            FILE *fpipe;

            sprintf(ns3_command, "%s %d %d %f %f %f %f %d\n", NS3_BIN, MLS.number_devices, MLS.simul_runs,
                MLS.population[thread_id].min_delay, MLS.population[thread_id].max_delay,
                MLS.population[thread_id].borders_threshold, MLS.population[thread_id].margin_forwarding,
                MLS.population[thread_id].neighbors_threshold);

            #ifndef NDEBUG
                fprintf(stderr, "[DEBUG][%d][%d] NS3 command line: %s\n", world_rank, thread_id, ns3_command);
            #endif

            if (!(fpipe = (FILE*)popen(ns3_command,"r")))
            {
                perror("Problems with pipe");
                exit(EXIT_FAILURE);
            }

            fscanf(fpipe, "%s", ns3_line);
            MLS.population[thread_id].energy = atof(ns3_line);

            fscanf(fpipe, "%s", ns3_line);
            MLS.population[thread_id].coverage = atof(ns3_line);

            fscanf(fpipe, "%s", ns3_line);
            MLS.population[thread_id].nforwardings = atof(ns3_line);

            fscanf(fpipe, "%s", ns3_line);
            MLS.population[thread_id].time = atof(ns3_line);

            pclose(fpipe);
            
            //#ifndef NDEBUG
            if ((world_rank < INIT_SOLS_COUNT)&&(thread_id == 0)) {
                fprintf(stderr, "[DEBUG][%d] Thread %d starting solution:\n", world_rank, thread_id);
                show_solution(&MLS.population[thread_id]);
            }
            //#endif

            // Envío la solución computada por la heurística a AGA.
            if ((MLS.population[thread_id].energy > 0) &&
                (MLS.population[thread_id].min_delay <= MLS.population[thread_id].max_delay) &&
                (MLS.population[thread_id].coverage >= MLS.min_coverage)) {

                pthread_mutex_lock(&MLS.mpi_mutex);
                    #ifndef NONMPI
                        #ifdef MPI_MODE_STANDARD
                            MPI_Send(&MLS.population[thread_id], 1, mpi_solution_type, AGA__PROCESS_RANK, AGA__NEW_SOL_MSG, MPI_COMM_WORLD);
                        #endif

                        #ifdef MPI_MODE_SYNC
                            MPI_Ssend(&MLS.population[thread_id], 1, mpi_solution_type, AGA__PROCESS_RANK, AGA__NEW_SOL_MSG, MPI_COMM_WORLD);
                        #endif

                        #ifdef MPI_MODE_BUFFERED
                            MPI_Bsend(&MLS.population[thread_id], 1, mpi_solution_type, AGA__PROCESS_RANK, AGA__NEW_SOL_MSG, MPI_COMM_WORLD);
                        #endif
                    #else
                        // NONMPI
                        MLS.population[thread_id].status = SOLUTION__STATUS_NEW;
                        for (int i = 0; (i < AGA__MAX_ARCHIVE_SIZE) && (MLS.population[thread_id].status == SOLUTION__STATUS_NEW); i++) {
                            if (AGA.population[i].status == SOLUTION__STATUS_EMPTY) {
                                // Encontré una posición libre. Agrego la solución al archivo acá.
                                clone_solution(&AGA.population[i], &MLS.population[thread_id]);
                                archivers_aga_add(i);
                                MLS.population[thread_id].status = SOLUTION__STATUS_READY;
                            }
                        }
                    #endif
                pthread_mutex_unlock(&MLS.mpi_mutex);
            }

            MLS.total_iterations[thread_id] = 0;

            // Comienza la busqueda.
            pthread_mutex_lock(&MLS.work_type_mutex[thread_id]);
                MLS.work_type[thread_id] = MLS__SEARCH;
            pthread_mutex_unlock(&MLS.work_type_mutex[thread_id]);
            
            work_type = MLS__SEARCH;
            
            pthread_barrier_wait(&MLS.sync_barrier);
        }
        else if (work_type == MLS__SEARCH) {
            double delta;
            double alfa = MLS.alpha;
            int rand_op;

            // =================================================================
            // Empiezo con la busqueda
            random = cpu_mt_generate(MLS.random_states[thread_id]);
            work_iteration_size = (int)(MLS__THREAD_FIXED_ITERS + (random * MLS__THREAD_RANDOM_ITERS));

            double min_delay;
            min_delay = MLS.population[thread_id].min_delay;

            double max_delay;
            max_delay = MLS.population[thread_id].max_delay;

            double borders_threshold;
            borders_threshold = MLS.population[thread_id].borders_threshold;

            double margin_forwarding;
            margin_forwarding = MLS.population[thread_id].margin_forwarding;

            int neighbors_threshold;
            neighbors_threshold = MLS.population[thread_id].neighbors_threshold;

            int rand_other_thread;
            rand_other_thread = (int)(cpu_mt_generate(MLS.random_states[thread_id]) * (MLS.count_threads-1));
            if (rand_other_thread >= thread_id) rand_other_thread++;

            MLS.total_iterations[thread_id]++;

            #ifndef NDEBUG
                fprintf(stderr, "[DEBUG][%d][%d] POS=%d %f %f %f %f %d\n", 
                    world_rank, thread_id, thread_id, 
                    MLS.population[thread_id].min_delay, 
                    MLS.population[thread_id].max_delay, 
                    MLS.population[thread_id].borders_threshold, 
                    MLS.population[thread_id].margin_forwarding, 
                    MLS.population[thread_id].neighbors_threshold);

                fprintf(stderr, "[DEBUG][%d][%d] POS=%d %f %f %f %f %d\n", 
                    world_rank, thread_id, rand_other_thread, 
                    MLS.population[rand_other_thread].min_delay, 
                    MLS.population[rand_other_thread].max_delay, 
                    MLS.population[rand_other_thread].borders_threshold, 
                    MLS.population[rand_other_thread].margin_forwarding, 
                    MLS.population[rand_other_thread].neighbors_threshold);
            #endif

            float prod = 0.0;
            for (search_iteration = 0; search_iteration < work_iteration_size; search_iteration++) {
                rand_op = cpu_mt_generate(MLS.random_states[thread_id]) * NUM_LS_OPERATORS;

                switch(rand_op){
                    case LS_ENERGY :
                    case LS_FORWARDING :
                        #ifndef NDEBUG
                            fprintf(stderr, "[DEBUG][%d][%d] LS_ENERGY && LS_FORWARDING\n", world_rank, thread_id);
                        #endif
                    
                        // Reduce borders_threshold
                        delta = MLS.population[rand_other_thread].borders_threshold - MLS.population[thread_id].borders_threshold;
                        
                        if (delta != 0) {
                            prod = alfa * delta;

                            if (delta > 0){
                                borders_threshold = borders_threshold-2*prod + 3*prod*cpu_mt_generate(MLS.random_states[thread_id]);
                            } else {
                                borders_threshold = borders_threshold+2*prod - 3*prod*cpu_mt_generate(MLS.random_states[thread_id]);
                            }

                            // Reduce neighbors_threshold
                            delta = MLS.population[rand_other_thread].neighbors_threshold - MLS.population[thread_id].neighbors_threshold;

                            if (delta > 0){
                                neighbors_threshold = floor(neighbors_threshold-2*prod + 3*prod*cpu_mt_generate(MLS.random_states[thread_id]));
                            } else {
                                neighbors_threshold = floor(neighbors_threshold+2*prod - 3*prod*cpu_mt_generate(MLS.random_states[thread_id]));
                            }
                        } else {
                            borders_threshold += ((borders_threshold * (alfa / 2) * cpu_mt_generate(MLS.random_states[thread_id])) - (alfa / 4));
                            neighbors_threshold += ((neighbors_threshold * (alfa / 2) * cpu_mt_generate(MLS.random_states[thread_id])) - (alfa / 4));
                        }

                        /*
                        if ((world_rank == 1)&&(thread_id == 0)) {
                            fprintf(stderr, "   >> LS_ENERGY || LS_FORWARDING: borders_threshold %.4f\n", borders_threshold);
                            fprintf(stderr, "   >> LS_ENERGY || LS_FORWARDING: neighbors_threshold %d\n", neighbors_threshold);
                        }*/
                        
                        if (borders_threshold < MLS.lbound_border_threshold)
                            borders_threshold = MLS.lbound_border_threshold;

                        if (borders_threshold > MLS.ubound_border_threshold)
                            borders_threshold = MLS.ubound_border_threshold;
                        
                        if (neighbors_threshold < MLS.lbound_neighbors_threshold)
                            neighbors_threshold = MLS.lbound_neighbors_threshold;

                        if (neighbors_threshold > MLS.ubound_neighbors_threshold)
                            neighbors_threshold = MLS.ubound_neighbors_threshold;

                        break;
                    case LS_COVERAGE :
                        #ifndef NDEBUG
                            fprintf(stderr, "[DEBUG][%d][%d] LS_COVERAGE\n", world_rank, thread_id);
                        #endif
                    
                        // Augment neighbors_threshold
                        delta = MLS.population[rand_other_thread].neighbors_threshold - MLS.population[thread_id].neighbors_threshold;

                        if (delta != 0) {
                            prod = alfa * delta;
                            
                            if (delta > 0){
                                neighbors_threshold = floor(neighbors_threshold-2*prod + 3*prod*cpu_mt_generate(MLS.random_states[thread_id]));
                            } else {
                                neighbors_threshold = floor(neighbors_threshold+2*prod - 3*prod*cpu_mt_generate(MLS.random_states[thread_id]));
                            }
                        } else {
                            neighbors_threshold += ((neighbors_threshold * (alfa / 2) * cpu_mt_generate(MLS.random_states[thread_id])) - (alfa / 4));
                        }

                        if (neighbors_threshold < MLS.lbound_neighbors_threshold)
                            neighbors_threshold = MLS.lbound_neighbors_threshold;

                        if (neighbors_threshold > MLS.ubound_neighbors_threshold)
                            neighbors_threshold = MLS.ubound_neighbors_threshold;
                        /*
                        if ((world_rank == 1)&&(thread_id == 0)) {
                            fprintf(stderr, "   >> LS_COVERAGE: neighbors_threshold %d\n", neighbors_threshold);
                        }*/

                        break;
                    case LS_TIME :
                        #ifndef NDEBUG
                            fprintf(stderr, "[DEBUG][%d][%d] LS_TIME\n", world_rank, thread_id);
                        #endif
                    
                        delta = MLS.population[thread_id].max_delay - MLS.population[thread_id].min_delay;

                        if (delta != 0) {
                            prod = alfa * delta;
                            
                            if (cpu_mt_generate(MLS.random_states[thread_id]) < 0.5) {
                                // Reduce max delay
                                max_delay = max_delay-2*prod + 3*prod*cpu_mt_generate(MLS.random_states[thread_id]);
                            } else {
                                // Augment min delay
                                min_delay = min_delay+2*prod - 3*prod*cpu_mt_generate(MLS.random_states[thread_id]);
                            }
                        } else {
                            if (cpu_mt_generate(MLS.random_states[thread_id]) < 0.5) {
                                max_delay += ((max_delay * (alfa / 2) * cpu_mt_generate(MLS.random_states[thread_id])) - (alfa / 4));
                            } else {
                                min_delay += ((min_delay * (alfa / 2) * cpu_mt_generate(MLS.random_states[thread_id])) - (alfa / 4));
                            }
                        }

                        if (min_delay < MLS.lbound_min_delay) min_delay = MLS.lbound_min_delay;
                        if (min_delay > MLS.ubound_min_delay) min_delay = MLS.ubound_min_delay;

                        if (max_delay < MLS.lbound_max_delay) max_delay = MLS.lbound_max_delay;
                        if (max_delay > MLS.ubound_max_delay) max_delay = MLS.ubound_max_delay;
                        /*
                        if ((world_rank == 1)&&(thread_id == 0)) {
                            fprintf(stderr, "   >> LS_TIME: min_delay %.4f\n", min_delay);
                            fprintf(stderr, "   >> LS_TIME: max_delay %.4f\n", max_delay);
                        }*/

                        break;
                }
            }

            double energy;
            double coverage;
            double nforwardings;
            double time;

            #ifndef ENABLE_HALF_SIMUL_PROBE
                sprintf(ns3_command, "%s %d %d %f %f %f %f %d\n", NS3_BIN, MLS.number_devices, MLS.simul_runs,
                    min_delay, max_delay, borders_threshold, margin_forwarding, neighbors_threshold);

                #ifndef NDEBUG
                    fprintf(stderr, "[DEBUG][%d][%d] NS3 command line: %s\n", world_rank, thread_id, ns3_command);
                #endif

                FILE *fpipe;

                if (!(fpipe = (FILE*)popen(ns3_command,"r")))
                {
                    perror("Problems with pipe");
                    exit(EXIT_FAILURE);
                }

                fscanf(fpipe, "%s", ns3_line);
                energy = atof(ns3_line);

                fscanf(fpipe, "%s", ns3_line);
                coverage = atof(ns3_line);

                fscanf(fpipe, "%s", ns3_line);
                nforwardings = atof(ns3_line);

                fscanf(fpipe, "%s", ns3_line);
                time = atof(ns3_line);

                pclose(fpipe);
            #else
                sprintf(ns3_command, "%s %d %d %f %f %f %f %d\n", NS3_BIN, MLS.number_devices, half_simul_runs,
                    min_delay, max_delay, borders_threshold, margin_forwarding, neighbors_threshold);

                #ifndef NDEBUG
                    fprintf(stderr, "[DEBUG][%d][%d] NS3 command line: %s\n", world_rank, thread_id, ns3_command);
                #endif

                FILE *fpipe;
                
                if (!(fpipe = (FILE*)popen(ns3_command,"r")))
                {
                    perror("Problems with pipe");
                    exit(EXIT_FAILURE);
                }

                double energy_1;
                double coverage_1;
                double nforwardings_1;
                double time_1;

                fscanf(fpipe, "%s", ns3_line);
                energy_1 = atof(ns3_line);

                fscanf(fpipe, "%s", ns3_line);
                coverage_1 = atof(ns3_line);

                fscanf(fpipe, "%s", ns3_line);
                nforwardings_1 = atof(ns3_line);

                fscanf(fpipe, "%s", ns3_line);
                time_1 = atof(ns3_line);

                pclose(fpipe);
                
                if ((time_1 < 2) && (energy_1 > 0) && (coverage_1 >= MLS.min_coverage)) {
                    bool is_non_dominated;
                    is_non_dominated = (MLS.population[thread_id].energy >= energy_1) ||
                        (MLS.population[thread_id].coverage <= coverage_1) ||
                        (MLS.population[thread_id].nforwardings >= nforwardings_1);
                
                    if ((is_non_dominated) || (!MLS.elite)) {
                        FILE *fpipe;

                        #ifndef NDEBUG
                            fprintf(stderr, "[DEBUG][%d][%d] Second run\n", world_rank, thread_id);
                        #endif

                        if (!(fpipe = (FILE*)popen(ns3_command,"r")))
                        {
                            perror("Problems with pipe");
                            exit(EXIT_FAILURE);
                        }

                        double energy_2;
                        double coverage_2;
                        double nforwardings_2;
                        double time_2;

                        fscanf(fpipe, "%s", ns3_line);
                        energy_2 = atof(ns3_line);

                        fscanf(fpipe, "%s", ns3_line);
                        coverage_2 = atof(ns3_line);

                        fscanf(fpipe, "%s", ns3_line);
                        nforwardings_2 = atof(ns3_line);

                        fscanf(fpipe, "%s", ns3_line);
                        time_2 = atof(ns3_line);

                        pclose(fpipe);
                        
                        energy = (energy_1 + energy_2) / 2;
                        coverage = (coverage_1 + coverage_2) / 2;
                        nforwardings = (nforwardings_1 + nforwardings_2) / 2;
                        time = (time_1 + time_2) / 2;
                        
                        #ifndef NDEBUG
                            fprintf(stderr, "[DEBUG][%d][%d] Population accepted\n", world_rank, thread_id);
                        #endif
                    } else {
                        #ifndef NDEBUG
                            fprintf(stderr, "[DEBUG][%d][%d] Population discarded (dominated)\n", world_rank, thread_id);
                        #endif
                        
                        energy = energy_1;
                        coverage = coverage_1;
                        nforwardings = nforwardings_1;
                        time = time_1;
                    }
                } else {
                    #ifndef NDEBUG
                        fprintf(stderr, "[DEBUG][%d][%d] Population discarded (out of bounds)\n", world_rank, thread_id);
                    #endif

                    energy = energy_1;
                    coverage = coverage_1;
                    nforwardings = nforwardings_1;
                    time = time_1;
                }
            #endif
            
            #ifndef NDEBUG
                fprintf(stderr, "[DEBUG][%d][%d] Result: %.3f  %.3f  %.3f  %.3f\n", world_rank, thread_id, 
                    energy, coverage, nforwardings, time);
            #endif

            if ((time < 2) && (energy > 0) && (coverage >= MLS.min_coverage)) {
                bool is_non_dominated;
                is_non_dominated = (MLS.population[thread_id].energy >= energy) ||
                    (MLS.population[thread_id].coverage <= coverage) ||
                    (MLS.population[thread_id].nforwardings >= nforwardings);
                
                if ((is_non_dominated) || (!MLS.elite)) {
                    MLS.population[thread_id].min_delay = min_delay;
                    MLS.population[thread_id].max_delay = max_delay;
                    MLS.population[thread_id].borders_threshold = borders_threshold;
                    MLS.population[thread_id].margin_forwarding = margin_forwarding;
                    MLS.population[thread_id].neighbors_threshold = neighbors_threshold;
                    MLS.population[thread_id].energy = energy;
                    MLS.population[thread_id].coverage = coverage;
                    MLS.population[thread_id].nforwardings = nforwardings;
                    MLS.population[thread_id].time = time;
                    /*
                    if ((world_rank == 1)&&(thread_id == 0)) {
                        fprintf(stderr, "[DEBUG] Resulting solution\n");
                        show_solution(&MLS.population[thread_id]);
                    }
                    */
                    /*
                    #ifndef NDEBUG
                        fprintf(stderr, "[DEBUG][%d] Thread %d found:\n", world_rank, thread_id);
                        show_solution(&MLS.population[thread_id]);
                    #endif*/

                    #ifndef NDEBUG
                        fprintf(stderr, "[DEBUG][%d][%d] Population accepted and sended to AGA\n", world_rank, thread_id);
                    #endif

                    pthread_mutex_lock(&MLS.mpi_mutex);
                        #ifndef NONMPI
                            #ifdef MPI_MODE_STANDARD
                                MPI_Send(&MLS.population[thread_id], 1, mpi_solution_type, AGA__PROCESS_RANK, AGA__NEW_SOL_MSG, MPI_COMM_WORLD);
                            #endif

                            #ifdef MPI_MODE_SYNC
                                MPI_Ssend(&MLS.population[thread_id], 1, mpi_solution_type, AGA__PROCESS_RANK, AGA__NEW_SOL_MSG, MPI_COMM_WORLD);
                            #endif

                            #ifdef MPI_MODE_BUFFERED
                                MPI_Bsend(&MLS.population[thread_id], 1, mpi_solution_type, AGA__PROCESS_RANK, AGA__NEW_SOL_MSG, MPI_COMM_WORLD);
                            #endif
                        #else
                            // NONMPI
                            MLS.population[thread_id].status = SOLUTION__STATUS_NEW;
                            for (int i = 0; (i < AGA__MAX_ARCHIVE_SIZE) && (MLS.population[thread_id].status == SOLUTION__STATUS_NEW); i++) {
                                if (AGA.population[i].status == SOLUTION__STATUS_EMPTY) {
                                    // Encontré una posición libre. Agrego la solución al archivo acá.
                                    clone_solution(&AGA.population[i], &MLS.population[thread_id]);
                                    archivers_aga_add(i);
                                    MLS.population[thread_id].status = SOLUTION__STATUS_READY;
                                }
                            }
                        #endif
                    pthread_mutex_unlock(&MLS.mpi_mutex);
                } else {
                    #ifndef NDEBUG
                        fprintf(stderr, "[DEBUG][%d][%d] Ignored because discarded (ELITE)\n", world_rank, thread_id);
                    #endif
                }
            } else {
                #ifndef NDEBUG
                    fprintf(stderr, "[DEBUG][%d][%d] Ignored because discarded (BOUNDS)\n", world_rank, thread_id);
                #endif
            }
        }

        if (MLS.total_iterations[thread_id] % reset_hit == 0) {
            pthread_barrier_wait(&MLS.sync_barrier);

            if (thread_id == 0) {
                #ifndef NDEBUG
                    fprintf(stderr, "[DEBUG] (%d) Refresh!\n", MLS.total_iterations[thread_id]);
                #endif

                #ifndef NONMPI
                    #ifndef NDEBUG
                        fprintf(stderr, "[DEBUG] MPI_Send\n");
                    #endif

                    #ifdef MPI_MODE_STANDARD
                        MPI_Send(&MLS.count_threads, 1, MPI_INT, AGA__PROCESS_RANK, AGA__REQ_SOL_MSG, MPI_COMM_WORLD);
                    #endif

                    #ifdef MPI_MODE_SYNC
                        MPI_Ssend(&MLS.count_threads, 1, MPI_INT, AGA__PROCESS_RANK, AGA__REQ_SOL_MSG, MPI_COMM_WORLD);
                    #endif

                    #ifdef MPI_MODE_BUFFERED
                        MPI_Bsend(&MLS.count_threads, 1, MPI_INT, AGA__PROCESS_RANK, AGA__REQ_SOL_MSG, MPI_COMM_WORLD);
                    #endif

                    #ifndef NDEBUG
                        fprintf(stderr, "[DEBUG] MPI_Recv\n");
                    #endif
                    MPI_Recv(MLS.population, 1, mpi_solution_type_array, AGA__PROCESS_RANK, AGA__REQ_SOL_MSG, MPI_COMM_WORLD, &status);

                    #ifndef NDEBUG
                        fprintf(stderr, "[DEBUG] ===================================================\n");
                        fprintf(stderr, "[DEBUG] [%d](%d) Population refresh\n", world_rank, MLS.total_iterations[thread_id]);
                        for (int i = 0; i < MLS.count_threads; i++) {
                            show_solution(&MLS.population[i]);
                        }
                        fprintf(stderr, "[DEBUG] ===================================================\n");
                    #endif
                #else
                    // ....
                #endif
            }

            pthread_barrier_wait(&MLS.sync_barrier);
        }
    }

    return NULL;
}
