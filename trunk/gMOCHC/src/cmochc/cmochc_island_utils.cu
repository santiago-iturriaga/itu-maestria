#include "cmochc_island_utils.h"

void population_init(int thread_id) {
    FLOAT random;
    
    for (int i = 0; i < MAX_POP_SOLS; i++) {
        // Random init.
        create_empty_solution(&(EA_THREADS[thread_id].population[i]));

        random = RAND_GENERATE(EA_INSTANCE.rand_state[thread_id]);
        
        int starting_pos;
        starting_pos = (int)(floor(INPUT.tasks_count * random));

        #ifdef DEBUG_3
            fprintf(stderr, "[DEBUG] Thread %d, inicializando solution %d, starting %d, direction %d...\n",
                thread_id, i, starting_pos, i & 0x1);
        #endif

        compute_mct_random(&(EA_THREADS[thread_id].population[i]), starting_pos, i & 0x1);

        if (i == 0) {
            EA_THREADS[thread_id].makespan_zenith_value = EA_THREADS[thread_id].population[i].makespan;
            EA_THREADS[thread_id].makespan_nadir_value = EA_THREADS[thread_id].makespan_zenith_value;
            //makespan_utopia_index = i;

            EA_THREADS[thread_id].energy_zenith_value = EA_THREADS[thread_id].population[i].energy_consumption;
            EA_THREADS[thread_id].energy_nadir_value = EA_THREADS[thread_id].energy_zenith_value;
            //energy_utopia_index = i;
        } else {
            #ifdef CMOCHC_LOCAL__MUTATE_INITIAL_POP
                mutate(EA_INSTANCE.rand_state[thread_id], &EA_THREADS[thread_id].population[i], &EA_THREADS[thread_id].population[i]);
            #endif

            if (EA_THREADS[thread_id].population[i].makespan < EA_THREADS[thread_id].makespan_zenith_value) {
                //makespan_utopia_index = i;
                EA_THREADS[thread_id].makespan_zenith_value = EA_THREADS[thread_id].population[i].makespan;

                if (EA_THREADS[thread_id].population[i].energy_consumption > EA_THREADS[thread_id].energy_nadir_value) {
                    EA_THREADS[thread_id].energy_nadir_value = EA_THREADS[thread_id].population[i].energy_consumption;
                }
            }
            if (EA_THREADS[thread_id].population[i].energy_consumption < EA_THREADS[thread_id].energy_zenith_value) {
                //energy_utopia_index = i;
                EA_THREADS[thread_id].energy_zenith_value = EA_THREADS[thread_id].population[i].energy_consumption;

                if (EA_THREADS[thread_id].population[i].makespan > EA_THREADS[thread_id].makespan_nadir_value) {
                    EA_THREADS[thread_id].makespan_nadir_value = EA_THREADS[thread_id].population[i].makespan;
                }
            }
        }
    }

    #ifdef DEBUG_3
        fprintf(stderr, "[DEBUG] Normalization references\n");
        fprintf(stderr, "> (makespan) zenith=%.2f, nadir=%.2f\n",
            EA_THREADS[thread_id].makespan_zenith_value, EA_THREADS[thread_id].makespan_nadir_value);
        fprintf(stderr, "> (energy) zenith=%.2f, nadir=%.2f\n",
            EA_THREADS[thread_id].energy_zenith_value, EA_THREADS[thread_id].energy_nadir_value);
    #endif

    for (int i = 0; i < MAX_POP_SOLS; i++) {
        EA_THREADS[thread_id].sorted_population[i] = i;
        EA_THREADS[thread_id].fitness_population[i] = NAN;
        fitness(thread_id, i);

        #ifdef DEBUG_3
            fprintf(stderr, "[DEBUG] Thread %d, solution=%d makespan=%.2f energy=%.2f fitness=%.2f\n",
                thread_id, i, EA_THREADS[thread_id].population[i].makespan, 
                EA_THREADS[thread_id].population[i].energy_consumption, fitness(thread_id, i));
        #endif
    }
}
