#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
//#include <cuda.h>
#include <omp.h>

#include "config.h"
#include "cuda-util.h"
#include "mtgp-1.1/mtgp32-cuda.h"
#include "billionga.h"

struct termination_criteria {
    int max_iteration_count;
};

inline void termination_criteria_init(struct termination_criteria *term_state,
    int max_iteration_count) {

    term_state->max_iteration_count = max_iteration_count;
}

inline int termination_criteria_eval(struct termination_criteria *term_state,
    struct bga_state *problem_state, int iteration_count) {

    return (iteration_count == term_state->max_iteration_count);
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stdout, "Wrong! RFM!\n\nUsage: %s <problem size> <max iteration> <prng vector size> <gpu device>\n(where 1 <= problem size <= %ld and problem_size can be divided by 8)\n\n", argv[0], LONG_MAX);
        return EXIT_FAILURE;
    }

    #if defined(INFO) || defined(DEBUG)
        fprintf(stdout, "[INFO] === Starting... ===============================\n");
    #endif

    long problem_size;
    problem_size = atol(argv[1]);

    int max_iteration_count = atoi(argv[2]);
    struct termination_criteria term_state;
    termination_criteria_init(&term_state, max_iteration_count);

    // === GPU.
    int number_gpus = 0;
    cudaGetDeviceCount(&number_gpus);
    if (number_gpus < 1)
    {
        fprintf(stderr, "[ERROR] No CUDA capable devices were detected.\n");
        exit(EXIT_FAILURE);
    }
    int starting_gpu_device = atoi(argv[4]);
    assert(starting_gpu_device >= 0 && starting_gpu_device < number_gpus);

    // === PRNG.
    int prng_vector_size = atoi(argv[3]);
    unsigned int prng_seeds[4] = {3822712292, 495793398, 4202624243, 3503457871}; // generated with: od -vAn -N4 -tu4 < /dev/urandom

    // === OpenMP
    int nthreads = omp_get_max_threads(); //omp_get_num_threads();
    //#if defined(INFO) || defined(DEBUG)
        fprintf(stdout, "[INFO] Number of threads %d.\n", nthreads);
    //#endif
    //assert(nthreads <= 4);
    assert(nthreads <= number_gpus);

    // === Inicialización del cGA
    struct bga_state problem_state;
    bga_initialization(&problem_state, problem_size, nthreads, NUMBER_OF_SAMPLES);

    #pragma omp parallel // private(th_id)
    {
        int current_iteration = 0;
        int th_id = omp_get_thread_num();

        int th_device = (starting_gpu_device + th_id) % number_gpus;
        ccudaSetDevice(th_device);

        #if defined(INFO) || defined(DEBUG)
            fprintf(stdout, "[INFO] Thread %d using device %d.\n", th_id, th_device);
        #endif

        assert(omp_get_num_threads() == nthreads);

        // === Inicialización del Mersenne Twister.
        mtgp32_status mt_status;
        mtgp32_initialize(&mt_status, prng_vector_size, prng_seeds[th_id]);

        // === Inicialización del BillionGA.
        bga_initialize_thread(&problem_state, th_id);

        #if defined(DEBUG)
            #pragma omp barrier
            if (th_id == 0) bga_show_prob_vector_state(&problem_state);
            #pragma omp barrier
        #endif

        long current_acc_prob = 0;

        while (!termination_criteria_eval(&term_state, &problem_state, current_iteration)) {           
            if (th_id == 0) {
                if (current_iteration % SHOW_UPDATE_EVERY == 0) {
                    fprintf(stdout, "*** ITERACION %d *********************************************\n", current_iteration);
                    
                    long aux;
                    aux = bga_get_part_accumulated_prob(&problem_state, th_id);

                    fprintf(stdout, "AUX = %ld, CURRENT = %ld, DIFF = %ld\n", aux, current_acc_prob, aux - current_acc_prob);
                    fprintf(stdout, "                  Value: %ld (improv: %ld)\n", aux, aux - current_acc_prob);
                    fprintf(stdout, "    Success probability: %.4f%%\n", (double)(aux * 100) / ((double)problem_state.max_prob_sum / nthreads));

                    current_acc_prob = aux;

                    aux = bga_get_part_stats_prob(&problem_state, th_id, 1, POPULATION_SIZE >> 1) * nthreads;
                    fprintf(stdout, " Aprox. prob. bit > 50%% (%d): %ld\n", POPULATION_SIZE >> 1, aux);
                    
                    aux = bga_get_part_stats_prob(&problem_state, th_id, -1, POPULATION_SIZE >> 1) * nthreads;
                    fprintf(stdout, " Aprox. prob. bit < 50%% (%d): %ld\n", POPULATION_SIZE >> 1, aux);

                    aux = bga_get_part_stats_prob(&problem_state, th_id, 1, (POPULATION_SIZE >> 1) + (POPULATION_SIZE >> 2)) * nthreads;
                    fprintf(stdout, "Aprox. prob. bit > 75%% (%d): %ld\n", (POPULATION_SIZE >> 1) + (POPULATION_SIZE >> 2), aux);
                    
                    aux = bga_get_part_stats_prob(&problem_state, th_id, -1, POPULATION_SIZE >> 2) * nthreads;
                    fprintf(stdout, "  Aprox. prob. bit < 25%% (%d): %ld\n", POPULATION_SIZE >> 2, aux);
                }
            }

            current_iteration++;

            bga_model_sampling_mt(&problem_state, &mt_status, th_id);

            #if defined(DEBUG)
                #pragma omp barrier
            #endif
            bga_compute_sample_part_fitness(&problem_state, th_id);

            #if defined(DEBUG)
                #pragma omp barrier
                if (th_id == 0) {
                    bga_show_samples(&problem_state);
                }
                #pragma omp barrier
            #endif

            #if defined(FULL_FITNESS_UPDATE)
                #pragma omp barrier
                if (th_id == 0) {
                    bga_compute_sample_full_fitness(&problem_state);
                }
                #pragma omp barrier
            #endif

            bga_model_update(&problem_state, th_id);
        }

        bga_get_part_accumulated_prob(&problem_state, th_id);
        long aux1[4], aux2[4], aux3[4], aux4[4];
        aux1[0] = aux2[0] = aux3[0] = aux4[0] = 0;
        aux1[1] = aux2[1] = aux3[1] = aux4[1] = 0;
        aux1[2] = aux2[2] = aux3[2] = aux4[2] = 0;
        aux1[3] = aux2[3] = aux3[3] = aux4[3] = 0;
        aux1[th_id] = bga_get_part_stats_prob(&problem_state, th_id, 1, POPULATION_SIZE >> 1);
        aux2[th_id] = bga_get_part_stats_prob(&problem_state, th_id, -1, POPULATION_SIZE >> 1);
        aux3[th_id] = bga_get_part_stats_prob(&problem_state, th_id, 1, (POPULATION_SIZE >> 1) + (POPULATION_SIZE >> 2));
        aux4[th_id] = bga_get_part_stats_prob(&problem_state, th_id, -1, POPULATION_SIZE >> 2);

        #pragma omp barrier
        if (th_id == 0) {
            long final_acc_prob = bga_get_full_accumulated_prob(&problem_state);
            fprintf(stdout, "*** FINAL *********************************************\n", current_iteration);
            fprintf(stdout, "                  Value: %ld\n", final_acc_prob);
            fprintf(stdout, "    Success probability: %.4f%%\n", (double)(final_acc_prob * 100) / (double)problem_state.max_prob_sum);
            fprintf(stdout, " Aprox. prob. bit > 50%% (%d): %ld\n", POPULATION_SIZE >> 1, aux1[0]+aux1[1]+aux1[2]+aux1[3]);
            fprintf(stdout, " Aprox. prob. bit < 50%% (%d): %ld\n", POPULATION_SIZE >> 1, aux2[0]+aux2[1]+aux2[2]+aux2[3]);
            fprintf(stdout, "Aprox. prob. bit > 75%% (%d): %ld\n", (POPULATION_SIZE >> 1) + (POPULATION_SIZE >> 2), aux3[0]+aux3[1]+aux3[2]+aux3[3]);
            fprintf(stdout, "  Aprox. prob. bit < 25%% (%d): %ld\n", POPULATION_SIZE >> 2, aux4[0]+aux4[1]+aux4[2]+aux4[3]);
        }

        // === Libero la memoria del Mersenne Twister.
        mtgp32_free(&mt_status);
    }

    // === Libero la memoria del cGA.
    bga_free(&problem_state);

    return EXIT_SUCCESS;
}
