#if !defined(CMOCHC_ISLANDS_UTILS__H)
#define CMOCHC_ISLANDS_UTILS__H

#include "../config.h"
#include "../solution.h"
#include "../load_params.h"
#include "../scenario.h"
#include "../etc_matrix.h"
#include "../energy_matrix.h"
#include "../utils.h"
#include "../basic/mct.h"
#include "../random/random.h"
#include "../archivers/aga.h"

#include "cmochc_island.h"
#include "cmochc_island_chc.h"

/* Ordena los elementos cmochc_island.weight_gap */
inline void gap_merge_sort() {
    int increment, l, l_max, r, r_max, current, i;

    increment = 1;
    int gap_length_r, gap_length_l;

    while (increment < EA_INSTANCE.weight_gap_count) {
        l = 0;
        r = increment;
        l_max = r - 1;
        r_max = (l_max + increment < EA_INSTANCE.weight_gap_count) ? l_max + increment : EA_INSTANCE.weight_gap_count - 1;

        current = 0;

        while (current < EA_INSTANCE.weight_gap_count) {
            while (l <= l_max && r <= r_max) {
                gap_length_r = EA_INSTANCE.weight_gap_length[EA_INSTANCE.weight_gap_sorted[r]];
                gap_length_l = EA_INSTANCE.weight_gap_length[EA_INSTANCE.weight_gap_sorted[l]];

                if (gap_length_r < gap_length_l) {
                    EA_INSTANCE.weight_gap_tmp[current] = EA_INSTANCE.weight_gap_sorted[r++];
                } else {
                    EA_INSTANCE.weight_gap_tmp[current] = EA_INSTANCE.weight_gap_sorted[l++];
                }

                current++;
            }

            while (r <= r_max) EA_INSTANCE.weight_gap_tmp[current++] = EA_INSTANCE.weight_gap_sorted[r++];
            while (l <= l_max) EA_INSTANCE.weight_gap_tmp[current++] = EA_INSTANCE.weight_gap_sorted[l++];

            l = r;
            r += increment;
            l_max = r - 1;
            r_max = (l_max + increment < EA_INSTANCE.weight_gap_count) ? l_max + increment : EA_INSTANCE.weight_gap_count - 1;
        }

        increment *= 2;

        for (i = 0; i < EA_INSTANCE.weight_gap_count; i++) {
            EA_INSTANCE.weight_gap_sorted[i] = EA_INSTANCE.weight_gap_tmp[i];
        }
    }
}

inline void merge_sort(struct solution *population, 
    FLOAT weight_makespan, FLOAT energy_makespan,
    FLOAT makespan_zenith_value, FLOAT energy_zenith_value,
    FLOAT makespan_nadir_value, FLOAT energy_nadir_value,
    int *sorted_population, FLOAT *fitness_population,
    int population_size, int *tmp) {

    int increment, l, l_max, r, r_max, current, i;

    increment = 1;
    FLOAT fitness_r, fitness_l;

    while (increment < population_size) {
        l = 0;
        r = increment;
        l_max = r - 1;
        r_max = (l_max + increment < population_size) ? l_max + increment : population_size - 1;

        current = 0;

        while (current < population_size) {
            while (l <= l_max && r <= r_max) {
                fitness_r = fitness(population, fitness_population, weight_makespan, energy_makespan,
                    makespan_zenith_value, energy_zenith_value, makespan_nadir_value, energy_nadir_value,
                    sorted_population[r]);
                fitness_l = fitness(population, fitness_population, weight_makespan, energy_makespan,
                    makespan_zenith_value, energy_zenith_value, makespan_nadir_value, energy_nadir_value,
                    sorted_population[l]);

                if (!isnan(fitness_r) && !isnan(fitness_l)) {
                    if (fitness_r < fitness_l) {
                        tmp[current] = sorted_population[r++];
                    } else {
                        tmp[current] = sorted_population[l++];
                    }
                } else if (!isnan(fitness_r) && isnan(fitness_l)) {
                    tmp[current] = sorted_population[r++];
                } else if (isnan(fitness_r) && !isnan(fitness_l)) {
                    tmp[current] = sorted_population[l++];
                } else {
                    /* Ambos son NAN, no importa */
                    tmp[current] = sorted_population[l++];
                }

                current++;
            }

            while (r <= r_max) tmp[current++] = sorted_population[r++];
            while (l <= l_max) tmp[current++] = sorted_population[l++];

            l = r;
            r += increment;
            l_max = r - 1;
            r_max = (l_max + increment < population_size) ? l_max + increment : population_size - 1;
        }

        increment *= 2;

        for (i = 0; i < population_size; i++) {
            sorted_population[i] = tmp[i];
        }
    }
}

#endif // CMOCHC_ISLANDS_UTILS__H
