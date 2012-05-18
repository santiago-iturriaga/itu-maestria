#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
//#include <cuda.h>
#include <omp.h>

#include "config.h"
#include "cuda-util.h"
#include "mtgp-1.1/mtgp32-cuda.h"
#include "billionga.h"

//Debe ser divisible entre 32 (8 y 4)... y 512, 128???
//#define TEST_PROBLEM_SIZE 32
//#define TEST_PROBLEM_SIZE 128
//#define TEST_PROBLEM_SIZE 524288
//#define TEST_PROBLEM_SIZE 1048576
//#define TEST_PROBLEM_SIZE 2097152
//#define TEST_PROBLEM_SIZE 899999744

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
    if(number_gpus < 1)
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
    assert(nthreads < 4);

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

        float current_acc_prob = bga_get_part_accumulated_prob(&problem_state, th_id);;

        while (!termination_criteria_eval(&term_state, &problem_state, current_iteration)) {
            current_iteration++;

            if (th_id == 0) {
                if (current_iteration % 1 == 0) {
                    fprintf(stdout, "*** ITERACION %d *********************************************\n", current_iteration);
                }
            }

            bga_model_sampling_mt(&problem_state, &mt_status, th_id);
            #if defined(DEBUG)
                #pragma omp barrier
            #endif
            bga_compute_sample_part_fitness(&problem_state, th_id);

            #if defined(FULL_FITNESS_UPDATE)
                #pragma omp barrier
                if (th_id == 0) {
                    bga_compute_sample_full_fitness(&problem_state);
                }
                #pragma omp barrier
            #endif
            
            bga_model_update(&problem_state, th_id);

            if (th_id == 0) {
                if (current_iteration % 1 == 0) {
                    float aux;                   
                    aux = bga_get_part_accumulated_prob(&problem_state, th_id);
                    
                    fprintf(stdout, "Accumulated probability: %.4f (delta: %f)\n", aux, aux - current_acc_prob);
                    current_acc_prob = aux;
                }
            }

            //#if defined(DEBUG)
            //if (th_id == 0) bga_show_prob_vector_state(&problem_state);
            //#endif

            #if !defined(DEBUG) && defined(INFO)
                if (!(termination_criteria_eval(&term_state, &problem_state, current_iteration))) {
                    bga_get_part_accumulated_prob(&problem_state, th_id);
                    
                    #pragma omp barrier
                    if (th_id == 0) {                       
                        if (current_iteration % 100 == 0) {

                            fprintf(stdout, "=== ITERACION %d ===============\n", current_iteration);
                            fprintf(stdout, "Accumulated probability: %.4f\n", bga_get_full_accumulated_prob(&problem_state));
                        }
                    }
                    #pragma omp barrier
                }
            #endif
        }

        bga_get_part_accumulated_prob(&problem_state, th_id);

        #pragma omp barrier
        if (th_id == 0) {
            float final_acc_prob = bga_get_full_accumulated_prob(&problem_state);
            fprintf(stdout, "\n\n[FINAL] Accumulated probability: %.4f\n", final_acc_prob);
            fprintf(stdout, "            Success probability: %.4f%%\n", final_acc_prob * 100 / problem_state.number_of_bits);
            
        }

        // === Libero la memoria del Mersenne Twister.
        mtgp32_free(&mt_status);
    }
    
    // === Libero la memoria del cGA.
    bga_free(&problem_state);

    return EXIT_SUCCESS;
}
