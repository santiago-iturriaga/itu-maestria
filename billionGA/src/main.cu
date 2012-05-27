#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
//#include <cuda.h>
#include <omp.h>
#include <time.h>

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
    struct bga_state *problem_state, int iteration_count, float fitness_sample_avg) {

    return ((iteration_count == term_state->max_iteration_count) || (fitness_sample_avg >= problem_state->max_prob_sum));
}

void timming_start(timespec &ts) {
    clock_gettime(CLOCK_REALTIME, &ts);
}

double timming_end(timespec &ts) {
    timespec ts_end;
    clock_gettime(CLOCK_REALTIME, &ts_end);

    double elapsed;
    elapsed = ((ts_end.tv_sec - ts.tv_sec) * 1000000.0) + ((ts_end.tv_nsec
            - ts.tv_nsec) / 1000.0);
    //fprintf(stdout, "[TIMMING] %s: %f microsegs.\n", message, elapsed);
    return elapsed;
}

int main(int argc, char **argv) {
    double cputime;
    float gputime;
    
    #if defined(MACRO_TIMMING)
        timespec full_start;
        timming_start(full_start);
    #endif
    
    if (argc != 5) {
        fprintf(stdout, "Wrong! RFM!\n\nUsage: %s <problem size> <max iteration> <prng vector size> <gpu device>\n(where 1 <= problem size <= %ld and problem_size can be divided by 8)\n\n", argv[0], LONG_MAX);
        return EXIT_FAILURE;
    }

    #if defined(INFO) || defined(DEBUG)
        fprintf(stdout, "[INFO] === Starting... ===============================\n");
    #endif

    #if defined(MACRO_TIMMING)
        timespec init1_start;
        timming_start(init1_start);
    #endif

    long problem_size;
    problem_size = atol(argv[1]);

    int max_iteration_count = atoi(argv[2]);
    struct termination_criteria term_state;
    termination_criteria_init(&term_state, max_iteration_count);

    fprintf(stdout, "[INFO] Cantidad de iteraciones %d.\n", max_iteration_count);

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

    #if defined(MACRO_TIMMING)
        cputime = timming_end(full_start);
        fprintf(stdout, "[TIME] Init (1) processing time: %f (microseconds)\n", cputime);
    #endif

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

        #if defined(MACRO_TIMMING)
            cudaEvent_t start;
            cudaEvent_t end;
            
            ccudaEventCreate(&start);
            ccudaEventCreate(&end);
            ccudaEventRecord(start, 0);
        #endif

        #if defined(INFO) || defined(DEBUG)
            fprintf(stdout, "[INFO] Thread %d using device %d.\n", th_id, th_device);
        #endif

        assert(omp_get_num_threads() == nthreads);

        // === Inicialización del Mersenne Twister.
        mtgp32_status mt_status;
        mtgp32_initialize(&mt_status, prng_vector_size, prng_seeds[th_id]);

        // === Inicialización del BillionGA.
        bga_initialize_thread(&problem_state, th_id);

        #if defined(MACRO_TIMMING)
            ccudaEventRecord(end, 0);
            ccudaEventSynchronize(end);
            ccudaEventElapsedTime(&gputime, start, end);
            fprintf(stdout, "[TIME] Init (2) processing time: %f (ms)\n", gputime);
        #endif

        #if defined(DEBUG)
            #pragma omp barrier
            if (th_id == 0) bga_show_prob_vector_state(&problem_state);
            #pragma omp barrier
        #endif

        long current_acc_prob = 0;
        long aux;
        float fitness_sample_avg = 0, probability_avg = 0, avg_fitness_porcentage = 0;
        long fitness_sample_a = 0, fitness_sample_b = 0;

        fprintf(stdout, "iter");
        #if defined(DEBUG)
            fprintf(stdout, ",avg. prob., abs. value,abs. improv.");
        #endif
        fprintf(stdout, ",f1,f2,avg f1 f2");
        #if defined(DEBUG)
            fprintf(stdout, ",gt 75,gt 50,lt 50,lt 25");
        #endif
        fprintf(stdout, "\n");
        
        while (!termination_criteria_eval(&term_state, &problem_state, current_iteration, fitness_sample_avg)) {                       
            if (th_id == 0) {
                if (current_iteration % SHOW_UPDATE_EVERY == 0) {
                                        
                    fprintf(stdout, "%d", current_iteration);
                    
                    #if defined(DEBUG)
                        aux = bga_get_full_accumulated_prob(&problem_state);
                        probability_avg = (float)(aux * 100.0 / problem_state.max_prob_sum);

                        fprintf(stdout, ",%.4f", probability_avg);
                        fprintf(stdout, ",%ld", aux);
                        fprintf(stdout, ",%ld", aux - current_acc_prob);
                    #endif
                    
                    avg_fitness_porcentage = (fitness_sample_avg / POPULATION_SIZE) * 100;
                    
                    fprintf(stdout, ",%ld,%ld,%.4f,%.4f", fitness_sample_a, fitness_sample_b, 
                        fitness_sample_avg, avg_fitness_porcentage);
                        
                    current_acc_prob = aux;

                    #if defined(DEBUG)
                        aux = bga_get_part_stats_prob(&problem_state, th_id, 1, (POPULATION_SIZE >> 1) + (POPULATION_SIZE >> 2)) * nthreads;
                        fprintf(stdout, ",%ld", aux);

                        aux = bga_get_part_stats_prob(&problem_state, th_id, 1, POPULATION_SIZE >> 1) * nthreads;
                        fprintf(stdout, ",%ld", aux);
                        
                        aux = bga_get_part_stats_prob(&problem_state, th_id, -1, POPULATION_SIZE >> 1) * nthreads;
                        fprintf(stdout, ",%ld", aux);
                        
                        aux = bga_get_part_stats_prob(&problem_state, th_id, -1, POPULATION_SIZE >> 2) * nthreads;                   
                        fprintf(stdout, ",%ld", aux);
                    #endif
                    fprintf(stdout, "\n");
                }
            }

            current_iteration++;

            #if defined(MACRO_TIMMING)
		if (current_iteration % SHOW_UPDATE_EVERY == 0) {
                	ccudaEventRecord(start, 0);
		}
            #endif

            bga_model_sampling_mt(&problem_state, &mt_status, th_id);

            #if defined(MACRO_TIMMING)
		if (current_iteration % SHOW_UPDATE_EVERY == 0) {
	                ccudaEventRecord(end, 0);
        	        ccudaEventSynchronize(end);
               		ccudaEventElapsedTime(&gputime, start, end);
                	fprintf(stdout, "[TIME] Sampling processing time: %f (ms)\n", gputime);
		}
            #endif

            #if defined(DEBUG)
                #pragma omp barrier
            #endif

            #if defined(MACRO_TIMMING)
		if (current_iteration % SHOW_UPDATE_EVERY == 0) {
                	ccudaEventRecord(start, 0);
		}
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

            #if defined(MACRO_TIMMING)
		if (current_iteration % SHOW_UPDATE_EVERY == 0) {
	                ccudaEventRecord(end, 0);
	                ccudaEventSynchronize(end);
	                ccudaEventElapsedTime(&gputime, start, end);
	                fprintf(stdout, "[TIME] Eval processing time: %f (ms)\n", gputime);
		}
            #endif

            #if defined(MACRO_TIMMING)
		if (current_iteration % SHOW_UPDATE_EVERY == 0) {
                	ccudaEventRecord(start, 0);
		}
            #endif

            bga_model_update(&problem_state, th_id);

            #if defined(MACRO_TIMMING)
		if (current_iteration % SHOW_UPDATE_EVERY == 0) {
	                ccudaEventRecord(end, 0);
	                ccudaEventSynchronize(end);
	                ccudaEventElapsedTime(&gputime, start, end);
	                fprintf(stdout, "[TIME] Update processing time: %f (ms)\n", gputime);
		}
            #endif
                        
            #if defined(FULL_FITNESS_UPDATE)
                fitness_sample_a = problem_state.samples_fitness[0];
                fitness_sample_b = problem_state.samples_fitness[1];
            #endif
            #if defined(PARTIAL_FITNESS_UPDATE)
                fitness_sample_a = problem_state.samples_vector_fitness[0][th_id];
                fitness_sample_b = problem_state.samples_vector_fitness[1][th_id];
            #endif
            
            fitness_sample_avg = (float)(fitness_sample_a + fitness_sample_b) / 2.0;
        }
       
        long aux0[4], aux1[4], aux2[4], aux3[4], aux4[4];
        
        aux1[0] = aux2[0] = aux3[0] = aux4[0] = 0;
        aux1[1] = aux2[1] = aux3[1] = aux4[1] = 0;
        aux1[2] = aux2[2] = aux3[2] = aux4[2] = 0;
        aux1[3] = aux2[3] = aux3[3] = aux4[3] = 0;
        
        aux1[th_id] = bga_get_part_stats_prob(&problem_state, th_id, 1, (POPULATION_SIZE >> 1) + (POPULATION_SIZE >> 2));
        aux2[th_id] = bga_get_part_stats_prob(&problem_state, th_id, 1, POPULATION_SIZE >> 1);
        aux3[th_id] = bga_get_part_stats_prob(&problem_state, th_id, -1, POPULATION_SIZE >> 1);
        aux4[th_id] = bga_get_part_stats_prob(&problem_state, th_id, -1, POPULATION_SIZE >> 2);

        bga_get_part_accumulated_prob(&problem_state, th_id); 

        #pragma omp barrier
        if (th_id == 0) {
            long final_acc_prob = bga_get_full_accumulated_prob(&problem_state);
            
            fprintf(stdout, ">>>>\n");
            fprintf(stdout, "iter,avg. prob., abs. value,abs. improv.,gt 75,gt 50,lt 50,lt 25\n");
            fprintf(stdout, "%d,%.4f,%ld,%ld,%ld,%ld,%ld\n",current_iteration, 
                (double)(final_acc_prob * 100) / (double)problem_state.max_prob_sum,
                final_acc_prob,
                aux1[0]+aux1[1]+aux1[2]+aux1[3],
                aux2[0]+aux2[1]+aux2[2]+aux2[3], 
                aux3[0]+aux3[1]+aux3[2]+aux3[3],
                aux4[0]+aux4[1]+aux4[2]+aux4[3]);
        }

        // === Libero la memoria del Mersenne Twister.
        mtgp32_free(&mt_status);
        
        #if defined(MACRO_TIMMING)
            ccudaEventDestroy(start);
            ccudaEventDestroy(end);
        #endif
    }
    
    #if defined(MACRO_TIMMING)
        cputime = timming_end(full_start);
        fprintf(stdout, "[TIME] Total processing time: %f (microseconds)\n", cputime);
    #endif

    // === Libero la memoria del cGA.
    bga_free(&problem_state);

    return EXIT_SUCCESS;
}
