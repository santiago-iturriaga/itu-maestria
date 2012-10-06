#include <pthread.h>
#include <math.h>
#include <stdlib.h>

#include "cmochc_island.h"
#include "cmochc_island_pals.h"
#include "cmochc_island_utils.h"
#include "cmochc_island_chc.h"

/* Statistics */
#ifdef DEBUG_1
    int COUNT_GENERATIONS[MAX_THREADS] = {0};

    /* Al menos una nueva solución fue agregada a la población de
     * padres del CHC del deme luego de aplicar una iteración de crossovers */
    int COUNT_AT_LEAST_ONE_CHILDREN_INSERTED[MAX_THREADS] = {0};
    /* Cantidad de veces que el CHC del deme mejoró la mejor solución que tenía */
    int COUNT_IMPROVED_BEST_SOL[MAX_THREADS] = {0};
    /* Cantidad de crossovers aplicados */
    int COUNT_CROSSOVER[MAX_THREADS] = {0};
    /* Cantidad de crossovers que produjeron al menos uno de los hijos
     * mejor a alguno de sus padres */
    int COUNT_IMPROVED_CROSSOVER[MAX_THREADS] = {0};
    /* Cantidad de soluciones mutadas en cataclismos */
    int COUNT_CATACLYSM[MAX_THREADS] = {0};
    /* Cantidad de soluciones mejoradas durante cataclismos */
    int COUNT_IMPOVED_CATACLYSM[MAX_THREADS] = {0};
    /* Cantidad de ejecuciones del método de migracion */
    int COUNT_MIGRATIONS[MAX_THREADS] = {0};
    /* Cantidad de soluciones migradas */
    int COUNT_SOLUTIONS_MIGRATED[MAX_THREADS] = {0};
    /* Cantidad histórica de ocurrencias de cada peso en el archivo AGA */
    int COUNT_HISTORIC_WEIGHTS[CMOCHC_PARETO_FRONT__PATCHES] = {0};
#endif

struct cmochc_island EA_INSTANCE;
struct cmochc_thread EA_THREADS[MAX_THREADS];

/* Inicializa los hilos y las estructuras de datos */
void init();

/* Obtiene los mejores elementos de cada población */
int gather();

/* Adapta los pesos de los threads */
int adapt_weights_mod_arm(RAND_STATE rstate);

/* Muestra el resultado de la ejecución */
void display_results();

/* Libera los recursos pedidos y finaliza la ejecución */
void finalize();

/* Logica de los esclavos */
void* slave_thread(void *data);

void compute_cmochc_island() {
    if (MAX_THREADS >= INPUT.thread_count) {
        fprintf(stderr, "[ERROR] Max. number of threads is %d (%d < 64)\n", MAX_THREADS, INPUT.thread_count);
    }

    // ==============================================================================
    // CPU CHC (islands)
    // ==============================================================================
    #if defined(DEBUG_0)
        fprintf(stderr, "[DEBUG] starting CPU CHC (islands)...\n");
    #endif

    // Timming -----------------------------------------------------
    TIMMING_START(ts_init);
    // Timming -----------------------------------------------------

    // Inicializo la memoria e inicializo los hilos de ejecucion.
    #if defined(DEBUG_1)
        fprintf(stderr, "[DEBUG] CPU CHC (islands): init\n");
    #endif

    init();

    #if defined(DEBUG_1)
        fprintf(stderr, " [OK]\n");
    #endif

    // Timming -----------------------------------------------------
    TIMMING_END(">> cmochc_init", ts_init);
    // Timming -----------------------------------------------------

    #if defined(DEBUG_1)
    int last_iter_sols_gathered = 0;
    #endif

    RAND_STATE rstate;
    RAND_INIT(INPUT.thread_count, rstate);

    int master_status = CMOCHC_MASTER_STATUS__CHC;
    int worker_status = 0;

    for (int iteracion = 0; iteracion < INPUT.max_iterations; iteracion++) {
        /* ******************************** */
        /* Espero que los esclavos terminen */
        /* ******************************** */
        
        pthread_mutex_lock(&EA_INSTANCE.status_cond_mutex);
            while (EA_INSTANCE.thread_idle_count < INPUT.thread_count) {                   
                pthread_cond_wait(&EA_INSTANCE.master_status_cond, &EA_INSTANCE.status_cond_mutex);
            }
        pthread_mutex_unlock(&EA_INSTANCE.status_cond_mutex);

        /* ************************************** */
        /* Proceso el resultado del estado actual */
        /* ************************************** */
        if (master_status == CMOCHC_MASTER_STATUS__CHC) {

            /* Incorporo las mejores soluciones al repositorio de soluciones */
            TIMMING_START(ts_gather);
            #if defined(DEBUG_3)
                fprintf(stderr, "[DEBUG] CPU CHC (islands): gather\n");
            #endif

            if (gather() > 0) {
                #if defined(DEBUG_1)
                last_iter_sols_gathered = iteracion;
                #endif
            }

            TIMMING_END(">> cmochc_gather", ts_gather);

        } else if (master_status == CMOCHC_MASTER_STATUS__LS) {
            
            // TODO: .....
            
        }

        /* ***************************** */
        /* Configuro el siguiente estado */
        /* ***************************** */
        if (iteracion + 1 >= INPUT.max_iterations) {

            /* Si esta es la última iteracion, les aviso a los esclavos */
            worker_status = CMOCHC_THREAD_STATUS__STOP;
            
        } else {

            master_status = CMOCHC_MASTER_STATUS__CHC;

            if (master_status == CMOCHC_MASTER_STATUS__CHC) {
                /* Re-configuro los pesos de búsqueda para el algoritmo CHC */
                TIMMING_START(ts_weights);

                #ifndef CMOCHC_PARETO_FRONT__FIXED_WEIGHTS
                    #ifdef CMOCHC_PARETO_FRONT__RANDOM_WEIGHTS
                        FLOAT random;

                        for (int i = 0; i < INPUT.thread_count; i++) {
                            random = RAND_GENERATE(rstate);
                            EA_INSTANCE.thread_weight_assignment[i] = (int)(random * CMOCHC_PARETO_FRONT__PATCHES);
                        }
                    #endif
                    #if defined(CMOCHC_PARETO_FRONT__ADAPT_AR_WEIGHTS) || defined(CMOCHC_PARETO_FRONT__ADAPT_AM_WEIGHTS)
                        adapt_weights_mod_arm(rstate);
                    #endif
                #endif

                TIMMING_END(">> ts_weights", ts_weights);
                
                worker_status = CMOCHC_THREAD_STATUS__CHC_FROM_ARCHIVE;
                
            } else if (master_status == CMOCHC_MASTER_STATUS__LS) {
                
                // TODO: .....
                
            }
        }
        
        /* ******************************************************* */
        /* Notifico a los esclavos que esta pronto el nuevo estado */
        /* ******************************************************* */
        
        pthread_mutex_lock(&EA_INSTANCE.status_cond_mutex);
            EA_INSTANCE.thread_idle_count = 0;

            for (int i = 0; i < INPUT.thread_count; i++) {
                EA_INSTANCE.thread_status[i] = worker_status;
            }

            pthread_cond_broadcast(&EA_INSTANCE.worker_status_cond);
        pthread_mutex_unlock(&EA_INSTANCE.status_cond_mutex);
       
    }

    RAND_FINALIZE(rstate);

    /* Bloqueo la ejecucion hasta que terminen todos los hilos. */
    for(int i = 0; i < INPUT.thread_count; i++)
    {
        if(pthread_join(EA_INSTANCE.threads[i], NULL))
        {
            fprintf(stderr, "Could not join thread %d\n", i);
            exit(EXIT_FAILURE);
        }
        else
        {
            #if defined(DEBUG_1)
                fprintf(stderr, "[DEBUG] thread %d <OK>\n", i);
            #endif
        }
    }

    /* Libero la memoria. */
    #if defined(DEBUG_1)
        fprintf(stderr, "[DEBUG] CPU CHC (islands): finalize\n");
    #endif

    display_results();

    #if defined(DEBUG_1)
        fprintf(stderr, "[DEBUG] Last solution gathered on iteration = %d\n", last_iter_sols_gathered);
    #endif

    finalize();
}

int adapt_weights_mod_arm(RAND_STATE rstate) {
    EA_INSTANCE.weight_gap_count = 0;
    int last_filled_patch = -1;

    for (int i = 0; i < CMOCHC_PARETO_FRONT__PATCHES; i++) {
        if (ARCHIVER.tag_count[i] > 0) {
            EA_INSTANCE.weight_gap_index[EA_INSTANCE.weight_gap_count] = i;
            EA_INSTANCE.weight_gap_length[EA_INSTANCE.weight_gap_count] = i - last_filled_patch - 1;
            EA_INSTANCE.weight_gap_sorted[EA_INSTANCE.weight_gap_count] = EA_INSTANCE.weight_gap_count;
            EA_INSTANCE.weight_gap_count++;

            last_filled_patch = i;
        }
    }

    if (CMOCHC_PARETO_FRONT__PATCHES > last_filled_patch + 1) {
        EA_INSTANCE.weight_gap_index[EA_INSTANCE.weight_gap_count] = CMOCHC_PARETO_FRONT__PATCHES;
        EA_INSTANCE.weight_gap_length[EA_INSTANCE.weight_gap_count] = CMOCHC_PARETO_FRONT__PATCHES - last_filled_patch - 1;
        EA_INSTANCE.weight_gap_sorted[EA_INSTANCE.weight_gap_count] = EA_INSTANCE.weight_gap_count;
        EA_INSTANCE.weight_gap_count++;
    }

    gap_merge_sort();

    #if defined(CMOCHC_PARETO_FRONT__ADAPT_AR_WEIGHTS)
        double random;
    #endif

    int sel_patch_idx = -1;
    int biggest_patch_index = EA_INSTANCE.weight_gap_sorted[EA_INSTANCE.weight_gap_count - 1];

    for (int t = 0; t < INPUT.thread_count; t++) {
        #if defined(CMOCHC_PARETO_FRONT__ADAPT_AR_WEIGHTS)
            if (EA_INSTANCE.weight_gap_length[biggest_patch_index] == 0) {
                sel_patch_idx = EA_INSTANCE.weight_gap_index[biggest_patch_index];
            } else if (EA_INSTANCE.weight_gap_length[biggest_patch_index] == 1) {
                sel_patch_idx = EA_INSTANCE.weight_gap_index[biggest_patch_index] - 1;
            } else {
                random = RAND_GENERATE(rstate);

                int random_length;
                random_length = EA_INSTANCE.weight_gap_length[biggest_patch_index] * random;

                sel_patch_idx = EA_INSTANCE.weight_gap_index[biggest_patch_index] - random_length - 1;
                if (sel_patch_idx < 0) sel_patch_idx = 0;
                if (sel_patch_idx >= CMOCHC_PARETO_FRONT__PATCHES) sel_patch_idx = CMOCHC_PARETO_FRONT__PATCHES - 1;
            }
        #endif
        #if defined(CMOCHC_PARETO_FRONT__ADAPT_AM_WEIGHTS)
            if (EA_INSTANCE.weight_gap_length[biggest_patch_index] == 0) {
                sel_patch_idx = EA_INSTANCE.weight_gap_index[biggest_patch_index];
            } else if (EA_INSTANCE.weight_gap_length[biggest_patch_index] == 1) {
                sel_patch_idx = EA_INSTANCE.weight_gap_index[biggest_patch_index] - 1;
            } else {
                sel_patch_idx = EA_INSTANCE.weight_gap_index[biggest_patch_index]
                    - (EA_INSTANCE.weight_gap_length[biggest_patch_index] / 2) - 1;
            }
        #endif

        assert(sel_patch_idx >= 0);
        assert(sel_patch_idx < CMOCHC_PARETO_FRONT__PATCHES);

        EA_INSTANCE.thread_weight_assignment[t] = sel_patch_idx;
        EA_INSTANCE.weight_thread_assignment[sel_patch_idx] = t;

        if (sel_patch_idx != EA_INSTANCE.weight_gap_index[biggest_patch_index]) {
            EA_INSTANCE.weight_gap_index[EA_INSTANCE.weight_gap_count] = sel_patch_idx;
            EA_INSTANCE.weight_gap_length[EA_INSTANCE.weight_gap_count] = EA_INSTANCE.weight_gap_length[biggest_patch_index]
                - EA_INSTANCE.weight_gap_index[biggest_patch_index] + sel_patch_idx;
            EA_INSTANCE.weight_gap_sorted[EA_INSTANCE.weight_gap_count] = EA_INSTANCE.weight_gap_count;

            assert(EA_INSTANCE.weight_gap_length[EA_INSTANCE.weight_gap_count] >= 0);

            EA_INSTANCE.weight_gap_count++;

            EA_INSTANCE.weight_gap_length[biggest_patch_index] =
                EA_INSTANCE.weight_gap_index[biggest_patch_index] - sel_patch_idx - 1;

            assert(EA_INSTANCE.weight_gap_length[biggest_patch_index] >= 0);
        }

        for (int j = EA_INSTANCE.weight_gap_count - 2; j < EA_INSTANCE.weight_gap_count; j++) {
            int pos = j;
            int aux;

            while ((pos > 0) && (EA_INSTANCE.weight_gap_length[EA_INSTANCE.weight_gap_sorted[pos]]
                <= EA_INSTANCE.weight_gap_length[EA_INSTANCE.weight_gap_sorted[pos-1]])) {

                aux = EA_INSTANCE.weight_gap_sorted[pos-1];
                EA_INSTANCE.weight_gap_sorted[pos-1] = EA_INSTANCE.weight_gap_sorted[pos];
                EA_INSTANCE.weight_gap_sorted[pos] = aux;

                pos--;
            }
        }

        biggest_patch_index = EA_INSTANCE.weight_gap_sorted[EA_INSTANCE.weight_gap_count - 1];
    }

    return 0;
}

/* Inicializa los hilos y las estructuras de datos */
void init() {
    fprintf(stderr, "[INFO] == CMOCHC/islands configuration constants ==============\n");
    fprintf(stderr, "       CMOCHC_LOCAL__POPULATION_SIZE               : %d\n", CMOCHC_LOCAL__POPULATION_SIZE);
    fprintf(stderr, "       CMOCHC_LOCAL__ITERATION_COUNT               : %d\n", CMOCHC_LOCAL__ITERATION_COUNT);
    fprintf(stderr, "       CMOCHC_LOCAL__BEST_SOLS_KEPT                : %d\n", CMOCHC_LOCAL__BEST_SOLS_KEPT);
    fprintf(stderr, "       CMOCHC_LOCAL__MATING_MAX_THRESHOLD_DIVISOR  : %d\n", CMOCHC_LOCAL__MATING_MAX_THRESHOLD_DIVISOR);
    fprintf(stderr, "       CMOCHC_LOCAL__MATING_THRESHOLD_STEP_DIVISOR : %d\n", CMOCHC_LOCAL__MATING_THRESHOLD_STEP_DIVISOR);

    fprintf(stderr, "       CMOCHC_LOCAL__MUTATE_INITIAL_POP            : ");
    #ifdef CMOCHC_LOCAL__MUTATE_INITIAL_POP
        fprintf(stderr, "YES\n");
    #else
        fprintf(stderr, "NO\n");
    #endif

    fprintf(stderr, "       CMOCHC_LOCAL__FITNESS_NORM                  : ");
    #ifdef CMOCHC_LOCAL__Z_FITNESS_NORM
        fprintf(stderr, "Z (zenith)\n");
    #endif
    #ifdef CMOCHC_LOCAL__ZN_FITNESS_NORM
        fprintf(stderr, "ZN (zenith/nadir)\n");
    #endif

    fprintf(stderr, "       CMOCHC_PARETO_FRONT__PATCHES                : %d\n", CMOCHC_PARETO_FRONT__PATCHES);
    fprintf(stderr, "       CMOCHC_PARETO_FRONT__ADAPT_WEIGHTS          : ");
    #ifdef CMOCHC_PARETO_FRONT__FIXED_WEIGHTS
        fprintf(stderr, "FIXED\n");
    #endif
    #ifdef CMOCHC_PARETO_FRONT__RANDOM_WEIGHTS
        fprintf(stderr, "RANDOM\n");
    #endif
    #ifdef CMOCHC_PARETO_FRONT__ADAPT_AR_WEIGHTS
        fprintf(stderr, "A-RANDOM\n");
    #endif
    #ifdef CMOCHC_PARETO_FRONT__ADAPT_AM_WEIGHTS
        fprintf(stderr, "A-MIDDLE\n");
    #endif

    fprintf(stderr, "       CMOCHC_COLLABORATION__MOEAD_NEIGH_SIZE      : %d\n", CMOCHC_COLLABORATION__MOEAD_NEIGH_SIZE);
    fprintf(stderr, "       CMOCHC_COLLABORATION__MIGRATION             : ");
    #ifdef CMOCHC_COLLABORATION__MIGRATION_BEST
        fprintf(stderr, "BEST\n");
    #endif
    #ifdef CMOCHC_COLLABORATION__MIGRATION_RANDOM_ELITE
        fprintf(stderr, "RANDOM_ELITE\n");
    #endif
    #ifdef CMOCHC_COLLABORATION__MIGRATION_NONE
        fprintf(stderr, "NONE\n");
    #endif

    fprintf(stderr, "       PALS__MAX_INTENTOS                          : %d\n", PALS__MAX_INTENTOS);    
    fprintf(stderr, "[INFO] ========================================================\n");

    /* Weights */
    assert(CMOCHC_PARETO_FRONT__PATCHES > 1);

    for (int patch_idx = 0; patch_idx < CMOCHC_PARETO_FRONT__PATCHES; patch_idx++) {
        EA_INSTANCE.weights[patch_idx] = (FLOAT)(patch_idx+1) / (FLOAT)(CMOCHC_PARETO_FRONT__PATCHES+1);
        EA_INSTANCE.weight_thread_assignment[patch_idx] = -1;

        #ifdef DEBUG_2
            fprintf(stderr, "[DEBUG] weights[%d] = (%.4f, %.4f)\n", patch_idx, EA_INSTANCE.weights[patch_idx], 1 - EA_INSTANCE.weights[patch_idx]);
        #endif
    }

    EA_INSTANCE.thread_idle_count = 0;

    if (INPUT.thread_count > 1) {
        for (int i = 0; i < INPUT.thread_count; i++) {
            EA_INSTANCE.thread_weight_assignment[i] = i * (CMOCHC_PARETO_FRONT__PATCHES / INPUT.thread_count);           
            //EA_THREADS[i].currently_assigned_weight = EA_INSTANCE.thread_weight_assignment[i];
            EA_INSTANCE.weight_thread_assignment[EA_INSTANCE.thread_weight_assignment[i]] = i;

            assert(EA_INSTANCE.thread_weight_assignment[i] < CMOCHC_PARETO_FRONT__PATCHES);

            #ifdef DEBUG_2
                fprintf(stderr, "[DEBUG] thread[%d] assigned to patch %d\n", i, EA_INSTANCE.thread_weight_assignment[i]);
            #endif

            EA_INSTANCE.thread_status[i] = CMOCHC_THREAD_STATUS__CHC_FROM_NEW;
        }
    } else {
        EA_INSTANCE.thread_weight_assignment[0] = CMOCHC_PARETO_FRONT__PATCHES / 2;
        //EA_THREADS[0].currently_assigned_weight = EA_INSTANCE.thread_weight_assignment[0];
        EA_INSTANCE.weight_thread_assignment[EA_INSTANCE.thread_weight_assignment[0]] = 0;

        assert(EA_INSTANCE.thread_weight_assignment[0] < CMOCHC_PARETO_FRONT__PATCHES);

        #ifdef DEBUG_2
            fprintf(stderr, "[DEBUG] thread[%d] assigned to patch %d\n", 0, EA_INSTANCE.thread_weight_assignment[0]);
        #endif

        EA_INSTANCE.thread_status[0] = CMOCHC_THREAD_STATUS__CHC_FROM_NEW;
    }

    /* Sync */
    if (pthread_barrier_init(&(EA_INSTANCE.sync_barrier), NULL, INPUT.thread_count + 1))
    {
        fprintf(stderr, "[ERROR] could not create sync_barrier.\n");
        exit(EXIT_FAILURE);
    }
    if (pthread_mutex_init(&(EA_INSTANCE.status_cond_mutex), NULL))
    {
        fprintf(stderr, "[ERROR] could not create status_cond_mutex.\n");
        exit(EXIT_FAILURE);
    }
    if (pthread_cond_init(&(EA_INSTANCE.worker_status_cond), NULL))
    {
        fprintf(stderr, "[ERROR] could not create worker_status_cond.\n");
        exit(EXIT_FAILURE);
    }
    if (pthread_cond_init(&(EA_INSTANCE.master_status_cond), NULL))
    {
        fprintf(stderr, "[ERROR] could not create master_status_cond.\n");
        exit(EXIT_FAILURE);
    }

    /* Inicializo el archivador */
    EA_INSTANCE.archiver_new_pop_size = INPUT.thread_count * CMOCHC_LOCAL__BEST_SOLS_KEPT;
    archivers_aga_init(CMOCHC_PARETO_FRONT__PATCHES);

    /* Inicializo los hilos */
    for (int i = 0; i < INPUT.thread_count; i++)
    {
        struct cmochc_thread *t_data;
        t_data = &(EA_THREADS[i]);
        t_data->thread_id = i;

        if (pthread_create(&(EA_INSTANCE.threads[i]), NULL, slave_thread, (void*) t_data))
        {
            fprintf(stderr, "[ERROR] could not create slave thread %d\n", i);
            exit(EXIT_FAILURE);
        }
    }
}

/* Obtiene los mejores elementos de cada población */
int gather() {
    int new_solutions;
    new_solutions = archivers_aga(EA_INSTANCE.archiver_new_pop_size);

    #ifdef DEBUG_1
        int current_tag;
        for (int s = 0; s < ARCHIVER__MAX_SIZE; s++) {
            if (ARCHIVER.population[s].initialized == SOLUTION__IN_USE) {
                current_tag = ARCHIVER.population_tag[s];
                COUNT_HISTORIC_WEIGHTS[current_tag]++;
            }
        }
    #endif

    return new_solutions;
}

void solution_migration(int thread_id) {
    FLOAT random;

    /* *********************************************************************************************
     * Busco las mejores soluciones elite a importar
     * ********************************************************************************************* */
    #ifndef CMOCHC_COLLABORATION__MIGRATION_NONE
        for (int i = 0; i < CMOCHC_COLLABORATION__MOEAD_NEIGH_SIZE; i++) {
            EA_THREADS[thread_id].migration_global_pop_index[i] = -1;
        }

        int ep_current_index, ep_solution_index;
        ep_current_index = 0;
        ep_solution_index = 0;

        #ifdef CMOCHC_COLLABORATION__MIGRATION_BEST
            int worst_distance = 0, worst_index = 0;
            int current_solution_distance;

            while ((ep_current_index < ARCHIVER__MAX_SIZE) &&
                (ep_solution_index < ARCHIVER.population_count)) {

                if (ARCHIVER.population[ep_current_index].initialized == SOLUTION__IN_USE) {
                    if (ep_solution_index < CMOCHC_COLLABORATION__MOEAD_NEIGH_SIZE) {
                        /* Aún no esta lleno el array con el vecindario */
                        EA_THREADS[thread_id].migration_global_pop_index[ep_solution_index] = ep_current_index;
                        EA_THREADS[thread_id].migration_current_weight_distance[ep_solution_index] =
                            abs(EA_INSTANCE.thread_weight_assignment[thread_id] -
                                ARCHIVER.population_tag[ep_current_index]);

                        if (ep_solution_index == 0) {
                            worst_distance = EA_THREADS[thread_id].migration_current_weight_distance[ep_solution_index];
                            worst_index = ep_solution_index;
                        } else {
                            if (worst_distance < EA_THREADS[thread_id].migration_current_weight_distance[ep_solution_index]) {
                                worst_distance = EA_THREADS[thread_id].migration_current_weight_distance[ep_solution_index];
                                worst_index = ep_solution_index;
                            }
                        }
                    } else {
                        current_solution_distance = abs(
                            EA_INSTANCE.thread_weight_assignment[thread_id] -
                            ARCHIVER.population_tag[ep_current_index]);

                        if (current_solution_distance < worst_distance) {
                            worst_distance = current_solution_distance;

                            EA_THREADS[thread_id].migration_global_pop_index[worst_index] = ep_current_index;
                            EA_THREADS[thread_id].migration_current_weight_distance[worst_index] = current_solution_distance;

                            for (int i = 0; i < CMOCHC_COLLABORATION__MOEAD_NEIGH_SIZE; i++) {
                                if (worst_distance < EA_THREADS[thread_id].migration_current_weight_distance[i]) {
                                    worst_distance = EA_THREADS[thread_id].migration_current_weight_distance[i];
                                    worst_index = i;
                                }
                            }
                        }
                    }

                    ep_solution_index++;
                }

                ep_current_index++;
            }
        #endif
        #ifdef CMOCHC_COLLABORATION__MIGRATION_RANDOM_ELITE
            int current_ep_solution_index;
            current_ep_solution_index = 0;

            FLOAT selection_prob;
            selection_prob = 1 / ARCHIVER.population_count;

            while ((ep_current_index < ARCHIVER__MAX_SIZE) &&
                (ep_solution_index < ARCHIVER.population_count) &&
                (current_ep_solution_index < CMOCHC_COLLABORATION__MOEAD_NEIGH_SIZE)) {

                if (ARCHIVER.population[ep_current_index].initialized == SOLUTION__IN_USE) {
                    if (RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]) <= selection_prob) {
                        /* Aún no esta lleno el array con el vecindario */
                        EA_THREADS[thread_id].migration_global_pop_index[current_ep_solution_index] = ep_current_index;
                        EA_THREADS[thread_id].migration_current_weight_distance[current_ep_solution_index] =
                            abs(EA_INSTANCE.thread_weight_assignment[thread_id] -
                                ARCHIVER.population_tag[ep_current_index]);

                        current_ep_solution_index++;
                    }

                    ep_solution_index++;
                }

                ep_current_index++;
            }
        #endif
    #endif

    /* *********************************************************************************************
     * Migro soluciones desde la población elite
     * ********************************************************************************************* */

    int migrated;

    int migrated_solution_index;
    migrated_solution_index = 0;

    int next_solution_index;
    next_solution_index = 0;

    while (next_solution_index < MAX_POP_SOLS) {
        migrated = 0;

        #ifndef CMOCHC_COLLABORATION__MIGRATION_NONE
            if (migrated_solution_index < CMOCHC_COLLABORATION__MOEAD_NEIGH_SIZE) {
                if (EA_THREADS[thread_id].migration_global_pop_index[migrated_solution_index] != -1) {
                    #if defined(DEBUG_1)
                        COUNT_MIGRATIONS[thread_id]++;
                    #endif

                    #ifdef CMOCHC_COLLABORATION__MIGRATE_BY_COPY
                        clone_solution(
                            &EA_THREADS[thread_id].population[next_solution_index],
                            &ARCHIVER.population[EA_THREADS[thread_id].migration_global_pop_index[migrated_solution_index]]);

                        next_solution_index++;
                        migrated_solution_index++;
                        #if defined(DEBUG_1)
                            COUNT_SOLUTIONS_MIGRATED[thread_id]++;
                        #endif
                        migrated = 1;
                    #endif
                    #ifdef CMOCHC_COLLABORATION__MIGRATE_BY_MATE
                        if (next_solution_index + 1 < MAX_POP_SOLS) {
                            random = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]);

                            int migration_parent_index;
                            migration_parent_index = (int)(random * next_solution_index);

                            FLOAT d = distance(
                                &ARCHIVER.population[EA_THREADS[thread_id].migration_global_pop_index[migrated_solution_index]],
                                &EA_THREADS[thread_id].population[migration_parent_index]);

                            if (d > EA_THREADS[thread_id].threshold_max) {
                                hux(EA_INSTANCE.rand_state[thread_id],
                                    &ARCHIVER.population[EA_THREADS[thread_id].migration_global_pop_index[migrated_solution_index]],
                                    &EA_THREADS[thread_id].population[migration_parent_index],
                                    &EA_THREADS[thread_id].population[EA_THREADS[thread_id].sorted_population[next_solution_index]],
                                    &EA_THREADS[thread_id].population[EA_THREADS[thread_id].sorted_population[next_solution_index+1]]);

                                next_solution_index += 2;
                                migrated_solution_index++;
                                #if defined(DEBUG_1)
                                    COUNT_SOLUTIONS_MIGRATED[thread_id] += 2;
                                #endif
                                migrated = 1;
                            }
                        }

                        if (migrated == 0) {
                            CHC__MUTATE(EA_INSTANCE.rand_state[thread_id],
                                &ARCHIVER.population[EA_THREADS[thread_id].migration_global_pop_index[migrated_solution_index]],
                                &EA_THREADS[thread_id].population[next_solution_index])

                            next_solution_index++;
                            migrated_solution_index++;
                            #if defined(DEBUG_1)
                                COUNT_SOLUTIONS_MIGRATED[thread_id]++;
                            #endif
                            migrated = 1;
                        }
                    #endif
                    #ifdef CMOCHC_COLLABORATION__MIGRATE_BY_MUTATE
                        MUTATE(EA_INSTANCE.rand_state[thread_id],
                            &ARCHIVER.population[EA_THREADS[thread_id].migration_global_pop_index[migrated_solution_index]],
                            &EA_THREADS[thread_id].population[next_solution_index])

                        next_solution_index++;
                        migrated_solution_index++;
                        #if defined(DEBUG_1)
                            COUNT_SOLUTIONS_MIGRATED[thread_id]++;
                        #endif

                        migrated = 1;
                    #endif
                }
            }
        #endif

        if (migrated == 0) {
            random = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]);

            CHC__MUTATE(EA_INSTANCE.rand_state[thread_id],
                &EA_THREADS[thread_id].population[(int)(random * next_solution_index)],
                &EA_THREADS[thread_id].population[next_solution_index])

            next_solution_index++;
        }
    }
}

/* Logica de los esclavos */
void* slave_thread(void *data) {
    struct cmochc_thread *t_data = (struct cmochc_thread*)data;
    int thread_id = t_data->thread_id;

    /* *********************************************************************************************
     * Inicializo el thread.
     * *********************************************************************************************/

    /* Inicialización del estado del generador aleatorio */
    RAND_INIT(thread_id, EA_INSTANCE.rand_state[thread_id]);

    /* Inicializo la busqueda local */
    pals_init(thread_id);

    EA_THREADS[thread_id].threshold_max = INPUT.tasks_count / CMOCHC_LOCAL__MATING_MAX_THRESHOLD_DIVISOR;
    EA_THREADS[thread_id].threshold_max = INPUT.tasks_count / CMOCHC_LOCAL__MATING_MAX_THRESHOLD_DIVISOR;
    EA_THREADS[thread_id].threshold_step = EA_THREADS[thread_id].threshold_max / CMOCHC_LOCAL__MATING_THRESHOLD_STEP_DIVISOR;
    if (EA_THREADS[thread_id].threshold_step == 0) EA_THREADS[thread_id].threshold_step = 1;

    int status = CMOCHC_THREAD_STATUS__IDLE;
    int currently_assigned_weight;
    
    while (status != CMOCHC_THREAD_STATUS__STOP) {
        /* ***************************** */
        /* Espero que me asignen trabajo */
        /* ***************************** */
        
        pthread_mutex_lock(&EA_INSTANCE.status_cond_mutex);
            while (EA_INSTANCE.thread_status[thread_id] == CMOCHC_THREAD_STATUS__IDLE) {               
                pthread_cond_wait(&EA_INSTANCE.worker_status_cond, &EA_INSTANCE.status_cond_mutex);
            }
            
            status = EA_INSTANCE.thread_status[thread_id];
        pthread_mutex_unlock(&EA_INSTANCE.status_cond_mutex);

        /* *********************************************************************************************
         * Actualizo la asignación de pesos y vecindad.
         * *********************************************************************************************/

        currently_assigned_weight = EA_INSTANCE.thread_weight_assignment[thread_id];
        EA_THREADS[thread_id].weight_makespan = EA_INSTANCE.weights[currently_assigned_weight];
        EA_THREADS[thread_id].weight_energy = 1 - EA_THREADS[thread_id].weight_makespan;
        
        /* ****************** */
        /* Proceso el trabajo */
        /* ****************** */            
        if ((status == CMOCHC_THREAD_STATUS__CHC_FROM_NEW)||(status == CMOCHC_THREAD_STATUS__CHC_FROM_ARCHIVE)) {
            
            if (status == CMOCHC_THREAD_STATUS__CHC_FROM_NEW) {
                
                /* *********************************************************************************************
                 * Inicializo la población.
                 * *********************************************************************************************/

                chc_population_init(thread_id);
                
            } else if (status == CMOCHC_THREAD_STATUS__CHC_FROM_ARCHIVE) {

                /* ********************************* */
                /* Migro soluciones desde el archivo */
                /* ********************************* */
                solution_migration(thread_id);

                /* ************************************************************ */
                /* Actualizo los puntos de normalización con la población local */
                /* ************************************************************ */
                EA_THREADS[thread_id].makespan_zenith_value = EA_THREADS[thread_id].population[0].makespan;
                EA_THREADS[thread_id].makespan_nadir_value = EA_THREADS[thread_id].population[0].makespan;
                EA_THREADS[thread_id].energy_zenith_value = EA_THREADS[thread_id].population[0].energy_consumption;
                EA_THREADS[thread_id].energy_nadir_value = EA_THREADS[thread_id].population[0].energy_consumption;

                for (int i = 1; i < MAX_POP_SOLS; i++) {
                    if (EA_THREADS[thread_id].population[i].makespan < EA_THREADS[thread_id].makespan_zenith_value) {
                        EA_THREADS[thread_id].makespan_zenith_value = EA_THREADS[thread_id].population[i].makespan;
                    }
                    if (EA_THREADS[thread_id].population[i].makespan > EA_THREADS[thread_id].makespan_nadir_value) {
                        EA_THREADS[thread_id].makespan_nadir_value = EA_THREADS[thread_id].population[i].makespan;
                    }

                    if (EA_THREADS[thread_id].population[i].energy_consumption < EA_THREADS[thread_id].energy_zenith_value) {
                        EA_THREADS[thread_id].energy_zenith_value = EA_THREADS[thread_id].population[i].energy_consumption;
                    }
                    if (EA_THREADS[thread_id].population[i].energy_consumption > EA_THREADS[thread_id].energy_nadir_value) {
                        EA_THREADS[thread_id].energy_nadir_value = EA_THREADS[thread_id].population[i].energy_consumption;
                    }
                }

                /* *********************************************************************************************
                 * Re-calculo el fitness de toda la población
                 *********************************************************************************************** */
                fitness_reset(thread_id);
                fitness_all(thread_id);
            }

            /* *********************************************************************************************
             * Sort the population
             *********************************************************************************************** */
             
            merge_sort(thread_id);
            
            /* *********************************************************************************************
             * CHC evolution
             * ********************************************************************************************* */
             
            chc_evolution(thread_id);
            
            /* *********************************************************************************************
             * Fin de la evolución
             * ********************************************************************************************* */

            /* Copio mis mejores soluciones a la población de intercambio principal */
            int new_sol_index;
            int local_best_index;

            for (int i = 0; i < CMOCHC_LOCAL__BEST_SOLS_KEPT; i++) {
                new_sol_index = thread_id * CMOCHC_LOCAL__BEST_SOLS_KEPT + i;
                local_best_index = EA_THREADS[thread_id].sorted_population[i];

                clone_solution(&ARCHIVER.new_solutions[new_sol_index], &EA_THREADS[thread_id].population[local_best_index]);
                ARCHIVER.new_solutions_tag[new_sol_index] = EA_INSTANCE.thread_weight_assignment[thread_id];
            }
            
        } else if (status == CMOCHC_THREAD_STATUS__LS) {
            
            // TODO: ............
            
        } 
        
        if (status != CMOCHC_THREAD_STATUS__STOP) {
            /* *************************** */
            /* Aviso al master que terminé */
            /* *************************** */
            
            pthread_mutex_lock(&EA_INSTANCE.status_cond_mutex);
                EA_INSTANCE.thread_idle_count++;

                EA_INSTANCE.thread_status[thread_id] = CMOCHC_THREAD_STATUS__IDLE;
                status = CMOCHC_THREAD_STATUS__IDLE;
                
                pthread_cond_signal(&EA_INSTANCE.master_status_cond);
            pthread_mutex_unlock(&EA_INSTANCE.status_cond_mutex);
        }
    }          

    // ================================================================
    // Finalizo el thread.
    // ================================================================

    /* Finalizo el generador de numeros aleatorios */
    RAND_FINALIZE(EA_INSTANCE.rand_state[thread_id]);
    
    /* Finalizo la búsqueda local */
    pals_free(thread_id);

    for (int p = 0; p < MAX_POP_SOLS; p++) {
        free_solution(&EA_THREADS[thread_id].population[p]);
    }

    return 0;
}

void display_results() {
    /* Show solutions */
    #if defined(OUTPUT_SOLUTION)
        archivers_aga_dump();
    #endif

    #ifdef DEBUG_1
        archivers_aga_show();
        
        FLOAT aux;
        for (int i = 0; i < ARCHIVER__MAX_SIZE; i++) {
            if (ARCHIVER.population[i].initialized == 1) {
                aux = ARCHIVER.population[i].machine_compute_time[0];
                for (int j = 1; j < INPUT.machines_count; j++) {
                    if (ARCHIVER.population[i].machine_compute_time[j] > aux) {
                        aux = ARCHIVER.population[i].machine_compute_time[j];
                    }
                }
                
                fprintf(stderr, "> %d state=%d makespan=%f(%f) energy=%f\n",
                    i, ARCHIVER.new_solutions[i].initialized,
                    ARCHIVER.population[i].makespan, aux,
                    ARCHIVER.population[i].energy_consumption);
            }
        }

        int count_generations = 0;
        int count_at_least_one_children_inserted = 0;
        int count_improved_best_sol = 0;
        int count_crossover = 0;
        int count_improved_crossover = 0;
        int count_cataclysm = 0;
        int count_improved_mutation = 0;
        int count_migrations = 0;
        int count_solutions_migrated = 0;
        int count_pals = 0;
        int count_pals_improv = 0;
        int count_pals_improv_move = 0;
        int count_pals_improv_swap = 0;
        int count_pals_decline = 0;
        int count_pals_decline_move = 0;
        int count_pals_decline_swap = 0;

        for (int t = 0; t < INPUT.thread_count; t++) {
            count_generations += COUNT_GENERATIONS[t];
            count_at_least_one_children_inserted += COUNT_AT_LEAST_ONE_CHILDREN_INSERTED[t];
            count_improved_best_sol += COUNT_IMPROVED_BEST_SOL[t];
            count_crossover += COUNT_CROSSOVER[t];
            count_improved_crossover += COUNT_IMPROVED_CROSSOVER[t];
            count_cataclysm += COUNT_CATACLYSM[t];
            count_improved_mutation += COUNT_IMPOVED_CATACLYSM[t];
            count_migrations += COUNT_MIGRATIONS[t];
            count_solutions_migrated += COUNT_SOLUTIONS_MIGRATED[t];
            count_pals += CHC_PALS_COUNT_EXECUTIONS[t];
            count_pals_improv += CHC_PALS_COUNT_FITNESS_IMPROV[t];
            count_pals_improv_move += CHC_PALS_COUNT_FITNESS_IMPROV_MOVE[t];
            count_pals_improv_swap += CHC_PALS_COUNT_FITNESS_IMPROV_SWAP[t];
            count_pals_decline += CHC_PALS_COUNT_FITNESS_DECLINE[t];
            count_pals_decline_move += CHC_PALS_COUNT_FITNESS_DECLINE_MOVE[t];
            count_pals_decline_swap += CHC_PALS_COUNT_FITNESS_DECLINE_SWAP[t];
        }

        int *count_pf_found;
        count_pf_found = (int*)(malloc(sizeof(int) * CMOCHC_PARETO_FRONT__PATCHES));

        for (int s = 0; s < CMOCHC_PARETO_FRONT__PATCHES; s++) {
            count_pf_found[s] = 0;
        }

        for (int s = 0; s < ARCHIVER__MAX_SIZE; s++) {
            if (ARCHIVER.population[s].initialized == SOLUTION__IN_USE) {
                count_pf_found[ARCHIVER.population_tag[s]]++;
            }
        }

        fprintf(stderr, "[INFO] == Statistics ==========================================\n");
        fprintf(stderr, "       count_generations                    : %d\n", count_generations);
        fprintf(stderr, "       count_at_least_one_children_inserted : %d (%.2f %%)\n", count_at_least_one_children_inserted,
            ((FLOAT)count_at_least_one_children_inserted/(FLOAT)count_generations)*100);
        fprintf(stderr, "       count_improved_best_sol              : %d (%.2f %%)\n", count_improved_best_sol,
            ((FLOAT)count_improved_best_sol/(FLOAT)count_generations)*100);
        fprintf(stderr, "       count_crossover                      : %d\n", count_crossover);
        fprintf(stderr, "       count_improved_crossover             : %d (%.2f %%)\n", count_improved_crossover,
            ((FLOAT)count_improved_crossover/(FLOAT)count_crossover)*100);
        fprintf(stderr, "       count_cataclysm                      : %d\n", count_cataclysm);
        fprintf(stderr, "       count_improved_mutation              : %d (%.2f %%)\n", count_improved_mutation,
            ((FLOAT)count_improved_mutation/(FLOAT)count_cataclysm)*100);
        fprintf(stderr, "       count_migrations                     : %d\n", count_migrations);
        fprintf(stderr, "       count_solutions_migrated             : %d (%.2f %%)\n", count_solutions_migrated,
            ((FLOAT)count_solutions_migrated/(FLOAT)count_migrations)*100);
        fprintf(stderr, "       count_pals                           : %d\n", count_pals);
        fprintf(stderr, "       count_pals_improvments               : %d (%.2f %%)\n", count_pals_improv,
            ((FLOAT)count_pals_improv/(FLOAT)count_pals)*100);
        fprintf(stderr, "       count_pals_improvments_move          : %d (%.2f %%)\n", count_pals_improv_move,
            ((FLOAT)count_pals_improv_move/(FLOAT)count_pals)*100);
        fprintf(stderr, "       count_pals_improvments_swap          : %d (%.2f %%)\n", count_pals_improv_swap,
            ((FLOAT)count_pals_improv_swap/(FLOAT)count_pals)*100);
        fprintf(stderr, "       count_pals_decline                   : %d (%.2f %%)\n", count_pals_decline,
            ((FLOAT)count_pals_decline/(FLOAT)count_pals)*100);
        fprintf(stderr, "       count_pals_decline_move              : %d (%.2f %%)\n", count_pals_decline_move,
            ((FLOAT)count_pals_decline_move/(FLOAT)count_pals)*100);
        fprintf(stderr, "       count_pals_decline_swap              : %d (%.2f %%)\n", count_pals_decline_swap,
            ((FLOAT)count_pals_decline_swap/(FLOAT)count_pals)*100);
                       
        fprintf(stderr, "       archive tag count:\n");
        for (int t = 0; t < CMOCHC_PARETO_FRONT__PATCHES; t++) {
            fprintf(stderr, "          [%d] = %d\n", t, ARCHIVER.tag_count[t]);
        }

        fprintf(stderr, "       historic archive tag count:\n");
        for (int t = 0; t < CMOCHC_PARETO_FRONT__PATCHES; t++) {
            fprintf(stderr, "          [%d] = %d (%d)\n", t, count_pf_found[t], COUNT_HISTORIC_WEIGHTS[t]);
        }

        fprintf(stderr, "[INFO] ========================================================\n");

        free(count_pf_found);
    #endif
}

/* Libera los recursos pedidos y finaliza la ejecución */
void finalize() {
    archivers_aga_free(EA_INSTANCE.archiver_new_pop_size);
    pthread_barrier_destroy(&(EA_INSTANCE.sync_barrier));
}
