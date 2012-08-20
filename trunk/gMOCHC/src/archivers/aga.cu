#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "aga.h"
#include "../config.h"
#include "../utils.h"

inline float get_objective(struct solution *s, int objective) {
    if (objective == 0) {
        return s->makespan;
    } else if (objective == 1) {
        return s->energy_consumption;
    } else {
        ASSERT(false)
    }
}

int archive_add_solution(struct aga_state *state, int new_solutions_count)
{
    // Given a solution s, add it to the archive if:
    //   a) the archive is empty
    //   b) the archive is not full and s is not dominated or equal to anything currently in the archive
    //   c) s dominates anything in the archive
    //   d) the archive is full but s is nondominated and is in a no more crowded square than at least one solution
    //      in addition, maintain the archive such that all solutions are nondominated.

    int rc = 0;

    /* Agrego las soluciones no dominadas en lo espacios vacíos */
    int s_pos = 0;
    int n_pos = 0;
    
    while ((new_solutions_count > 0)&&(state->population_count < state->population_size)) {
        ASSERT(n_pos < state->new_solutions_size);
        ASSERT(s_pos < state->population_size);
        
        if (state->population[n_pos].initialized == 0) {
            n_pos++;
        } else {
            if (state->population[s_pos].initialized == 1) {
                s_pos++;
            } else {
                if (state->population[s_pos].initialized == -1) {
                    create_empty_solution(&state->population[s_pos],
                        state->new_solutions[n_pos].s, 
                        state->new_solutions[n_pos].etc, 
                        state->new_solutions[n_pos].energy);
                }
                
                clone_solution(&state->population[s_pos], &state->new_solutions[n_pos]);
                
                // Marco la solución como ya creada e invalido la copia en la población de nuevas soluciones.
                state->new_solutions[n_pos].initialized = 0;
                
                // Cambio la asignación del square de la solución
                state->grid_sol_loc[s_pos] = state->grid_sol_loc[n_pos + state->population_size];
                
                new_solutions_count--;
                state->population_count++; 
                rc++;
            }
        }
    }

    n_pos = 0;

    while (new_solutions_count > 0)  // Some new solutions are non-dominated.
    {
        ASSERT(n_pos < state->new_solutions_size);
        
        if (state->new_solutions[n_pos].initialized == 0) {
            n_pos++;
        } else {
            int to_replace;
            to_replace = -1;

            int new_solution_grid_pos;
            new_solution_grid_pos = state->grid_sol_loc[n_pos + state->population_size];

            int most_square_pop;
            most_square_pop = state->grid_pop[new_solution_grid_pos];

            for (int i = 0; i < state->population_size; i++)
            {
                if (state->population[i].initialized == 1) {
                    int i_grid_pos;
                    i_grid_pos = state->grid_sol_loc[i];

                    int i_square_pop;
                    i_square_pop = state->grid_pop[i_grid_pos];

                    if (i_square_pop > most_square_pop)
                    {
                        most_square_pop = state->grid_pop[i_square_pop];
                        to_replace = i;
                    }
                }
            }

            if (to_replace != -1)
            {
                clone_solution(&state->population[to_replace], &state->new_solutions[n_pos]);
                
                // Marco la solución como ya creada e invalido la copia en la población de nuevas soluciones.
                state->new_solutions[n_pos].initialized = 0;

                // Decremento la población en el square de la solución que acabo de borrar
                state->grid_pop[state->grid_sol_loc[to_replace]]--;
                
                // Cambio la asignación del square de la solución
                state->grid_sol_loc[to_replace] = state->grid_sol_loc[n_pos + state->population_size];

                state->population_count++;
                rc++;
            }
            else
            {
                // Marco la solución como ya creada e invalido la copia en la población de nuevas soluciones.
                state->new_solutions[n_pos].initialized = 0;

                // Decremento la población en el square de la solución que acabo de borrar
                state->grid_pop[state->grid_sol_loc[n_pos + state->population_size]]--;
            }
            
            new_solutions_count--;
        }
    }

    /*
    double actual_best_makespan = INFINITY, actual_best_energy = INFINITY;

    for (int i = 0; i < instance->population_max_size; i++) {
        if (instance->population[i].status == SOLUTION__STATUS_READY) {
            if (actual_best_makespan > get_makespan(&(instance->population[i]))) {
                actual_best_makespan = get_makespan(&(instance->population[i]));
            }

            if (actual_best_energy > get_energy(&(instance->population[i]))) {
                actual_best_energy = get_energy(&(instance->population[i]));
            }
        }
    }
    * */

    //fprintf(stdout, ">>> Min makespan(%f) Min energy(%f)\n", actual_best_makespan, actual_best_energy);
    return rc;
}

inline int find_loc(struct aga_state *state, int solution_pos)
{
    // Find the grid location of a solution given a vector of its objective values.
    int loc = 0;

    int d;
    int n = 1;

    int i;

    int inc[ARCHIVER__OBJECTIVES];
    double width[ARCHIVER__OBJECTIVES];

    // if the solution is out of range on any objective, return 1 more than the maximum possible grid location number
    for (i = 0; i < ARCHIVER__OBJECTIVES; i++)
    {
        if (solution_pos < state->population_size) {
            if ((get_objective(&(state->population[solution_pos]), i) < state->gl_offset[i]) ||
                (get_objective(&(state->population[solution_pos]), i) > state->gl_offset[i] + state->gl_range[i])) {

                return state->max_locations;
            }
        } else {
            if ((get_objective(&(state->new_solutions[solution_pos - state->population_size]), i) < state->gl_offset[i]) ||
                (get_objective(&(state->new_solutions[solution_pos - state->population_size]), i) > state->gl_offset[i] + state->gl_range[i])) {

                return state->max_locations;
            }
        }
    }

    for (i = 0; i < ARCHIVER__OBJECTIVES; i++)
    {
        inc[i] = n;
        n *=2;
        width[i] = state->gl_range[i];
    }

    for (d = 1; d <= ARCHIVER__AGA_DEPTH; d++)
    {
        for (i = 0; i < ARCHIVER__OBJECTIVES; i++)
        {
            if (solution_pos < state->population_size) {
                if(get_objective(&(state->population[solution_pos]), i) < 
                    width[i]/2 + state->gl_offset[i]) {
                        
                    loc += inc[i];
                } else {
                    state->gl_offset[i] += width[i]/2;
                }
            } else {
                if(get_objective(&(state->new_solutions[solution_pos - state->population_size]), i) < 
                    width[i]/2 + state->gl_offset[i]) {
                        
                    loc += inc[i];
                } else {
                    state->gl_offset[i] += width[i]/2;
                }
            }
        }
        for (i = 0; i < ARCHIVER__OBJECTIVES; i++)
        {
            inc[i] *= (ARCHIVER__OBJECTIVES *2);
            width[i] /= 2;
        }
    }

    return loc;
}

void update_grid(struct aga_state *state)
{
    // recalculate ranges for grid in the light of a new solution s
    int a, b;

    double offset[ARCHIVER__OBJECTIVES];
    double largest[ARCHIVER__OBJECTIVES];

    for (a = 0; a < ARCHIVER__OBJECTIVES; a++)
    {
        offset[a] = ARCHIVER__AGA_LARGE;
        largest[a] = -ARCHIVER__AGA_LARGE;
    }

    for (b = 0; b < ARCHIVER__OBJECTIVES; b++)
    {
        for (a = 0; a < state->population_size + state->new_solutions_size; a++)
        {
            if (a < state->population_size) {
                if (state->population[a].initialized != 0) {
                    if (get_objective(&(state->population[a]), b) < offset[b]) {
                        offset[b] = get_objective(&(state->population[a]), b);
                    }

                    if (get_objective(&(state->population[a]), b) > largest[b]) {
                        largest[b] = get_objective(&(state->population[a]), b);
                    }
                }
            } else {
                if (state->new_solutions[a - state->population_size].initialized != 0) {
                    if (get_objective(&(state->new_solutions[a - state->population_size]), b) < offset[b]) {
                        offset[b] = get_objective(&(state->new_solutions[a - state->population_size]), b);
                    }

                    if (get_objective(&(state->new_solutions[a - state->population_size]), b) > largest[b]) {
                        largest[b] = get_objective(&(state->new_solutions[a - state->population_size]), b);
                    }
                }
            }
        }
    }

    /*
    double sse = 0;
    double product = 1;

    for (a = 0; a < ARCHIVER__OBJECTIVES; a++)
    {
        sse += ((state->gl_offset[a] - offset[a]) * (state->gl_offset[a] - offset[a]));
        sse += ((state->gl_largest[a] - largest[a]) * (state->gl_largest[a] - largest[a]));
        product *= state->gl_range[a];
    }
    * */

    //static int change = 0;
    int square;

    // If the summed squared error (difference) between old and new
    // minima and maxima in each of the objectives
    // is bigger than 10 percent of the square of the size of the space
    // then renormalise the space and recalculte grid locations
    /*if (sse > (0.1 * product * product))
    {*/
        //change++;

        for (a = 0; a < ARCHIVER__OBJECTIVES; a++)
        {
            state->gl_largest[a] = largest[a] + 0.2 * largest[a];
            state->gl_offset[a] = offset[a] + 0.2 * offset[a];
            state->gl_range[a] = state->gl_largest[a] - state->gl_offset[a];
        }

        for (a = 0; a < state->max_locations; a++)
        {
            state->grid_pop[a] = 0;
        }

        for (a = 0; a < state->population_size + state->new_solutions_size; a++)
        {
            if (a < state->population_size) {
                if (state->population[a].initialized != 0) {
                    square = find_loc(state, a);

                    state->grid_sol_loc[a] = square;
                    state->grid_pop[square]++;
                }
            } else {
                if (state->new_solutions[a - state->new_solutions_size].initialized != 0) {
                    square = find_loc(state, a);

                    state->grid_sol_loc[a] = square;
                    state->grid_pop[square]++;
                }
            }
        }
    /*} else {
        for (a = 0; a < state->new_solutions_size; a++)
        {
            if (state->new_solutions[a].initialized != 0) {
                square = find_loc(state, a + state->population_size);

                state->grid_sol_loc[a + state->population_size] = square;
                state->grid_pop[square]++;
            }
        }
    }*/
}

void archivers_aga_init(struct aga_state *state, int population_max_size,
    struct solution *new_solutions, int new_solutions_size) {
        
    state->population = (struct solution*)(malloc(sizeof(struct solution) * population_max_size));
    state->population_size = population_max_size;
    state->population_count = 0;

    for (int p = 0; p < state->population_size; p++) {
        state->population[p].initialized = -1;
    }

    state->new_solutions = new_solutions;
    state->new_solutions_size = new_solutions_size;

    state->max_locations = pow(2, ARCHIVER__AGA_DEPTH * ARCHIVER__OBJECTIVES); // Number of locations in grid.
    state->grid_pop = (int*)(malloc(sizeof(int) * (state->max_locations + 1)));
    state->grid_sol_loc = (int*)(malloc(sizeof(int) * (state->population_size + new_solutions_size)));

    for (int i = 0; i < ARCHIVER__OBJECTIVES; i++) {
        state->gl_offset[i] = 0.0;
        state->gl_range[i] = 0.0;
        state->gl_largest[i] = 0.0;
    }

    for (int i = 0; i < state->max_locations; i++) {
        state->grid_pop[i] = 0;
    }
    
    #ifdef DEBUG_1
        fprintf(stderr, "[DEBUG] Archiver init\n");
        fprintf(stderr, "> population_size   : %d\n", state->population_size);
        fprintf(stderr, "> new_solutions_size: %d\n", state->new_solutions_size);
    #endif
}

void archivers_aga_free(struct aga_state *state) {
    free(state->population);
    free(state->grid_pop);
    free(state->grid_sol_loc);
}

int delete_dominated_solutions(struct aga_state *state) {   
    // Elimino las soluciones dominadas entre si de la población de nuevas soluciones.
    for (int n_pos = 0; n_pos < state->new_solutions_size; n_pos++)
    {
        if (state->new_solutions[n_pos].initialized == 1) 
        {
            for (int n2_pos = n_pos+1; (n2_pos < state->new_solutions_size) && 
                (state->new_solutions[n_pos].initialized == 1); n2_pos++)
            {
                if (state->new_solutions[n2_pos].initialized == 1) 
                {
                    // Calculo no dominancia del elemento nuevo con el actual.
                    if ((state->new_solutions[n2_pos].makespan <= state->new_solutions[n_pos].makespan) && 
                        (state->new_solutions[n2_pos].energy_consumption <= state->new_solutions[n_pos].energy_consumption))
                    {                       
                        state->new_solutions[n_pos].initialized = 0;
                    }
                    else if ((state->new_solutions[n_pos].makespan <= state->new_solutions[n2_pos].makespan) && 
                        (state->new_solutions[n_pos].energy_consumption <= state->new_solutions[n2_pos].energy_consumption))
                    {
                        state->new_solutions[n2_pos].initialized = 0;
                    }
                }
            }
        }
    }
    
    int cant_new_solutions = 0;
    
    // Comparo la dominancia de las nuevas soluciones con las viejas soluciones.
    for (int n_pos = 0; n_pos < state->new_solutions_size; n_pos++)
    {
        if (state->new_solutions[n_pos].initialized == 1) 
        {
            for (int s_pos = 0; (s_pos < state->population_size) && 
                (state->new_solutions[n_pos].initialized == 1); s_pos++)
            {
                if (state->population[s_pos].initialized == 1) 
                {
                    // Calculo no dominancia del elemento nuevo con el actual.
                    if ((state->population[s_pos].makespan <= state->new_solutions[n_pos].makespan) && 
                        (state->population[s_pos].energy_consumption <= state->new_solutions[n_pos].energy_consumption))
                    {
                        // La nueva solucion es dominada por una ya existente.                        
                        state->new_solutions[n_pos].initialized = 0;
                    }
                    else if ((state->new_solutions[n_pos].makespan <= state->population[s_pos].makespan) && 
                        (state->new_solutions[n_pos].energy_consumption <= state->population[s_pos].energy_consumption))
                    {
                        // La nueva solucion domina a una ya existente.
                        state->population[s_pos].initialized = 0;
                        state->population_count--;
                    }
                }
            }
        }
        
        if (state->new_solutions[n_pos].initialized == 1) 
        {
            cant_new_solutions++;
        }
    }
    
    return cant_new_solutions;
}

void archive_add_all_new(struct aga_state *state) {
    for (int n_pos = 0; n_pos < state->new_solutions_size; n_pos++)
    {
        if (state->new_solutions[n_pos].initialized == 1) {
            for (int s_pos = 0; (s_pos < state->population_size) && 
                (state->new_solutions[n_pos].initialized == 1); s_pos++)
            {
                if (state->population[s_pos].initialized != 1) 
                {
                    #ifdef DEBUG_3
                        fprintf(stderr, "> found empty space in %d\n", s_pos);
                    #endif
                    
                    if (state->population[s_pos].initialized == -1) {
                        create_empty_solution(&state->population[s_pos],
                            state->new_solutions[n_pos].s, 
                            state->new_solutions[n_pos].etc, 
                            state->new_solutions[n_pos].energy);
                    }
                    
                    clone_solution(&state->population[s_pos], &state->new_solutions[n_pos]);
                    state->population_count++;
                    state->new_solutions[n_pos].initialized = 0;
                }
            }
        }
    }
}

int archivers_aga(struct aga_state *state)
{
    #ifdef DEBUG_3
        fprintf(stderr, "[DEBUG] archiver aga...\n");
    #endif
    
    int new_solutions = delete_dominated_solutions(state);

    #ifdef DEBUG_3
        fprintf(stderr, "> solutions this iteration: %d\n", new_solutions);
    #endif

    #ifdef DEBUG_3
        fprintf(stderr, "> current solutions in archive: %d of %d\n", state->population_count, state->population_size);
    #endif    
    if (new_solutions + state->population_count < state->population_size) {
        #ifdef DEBUG_3
            fprintf(stderr, "> there is room in the archive for all gathered solutions\n");
        #endif    
        archive_add_all_new(state);
    } else {
        // Calculate grid location of mutant solution and renormalize archive if necessary.
        update_grid(state);

        // Update the archive by removing all dominated individuals.
        new_solutions = archive_add_solution(state, new_solutions);
    }
    
    return new_solutions;
}
