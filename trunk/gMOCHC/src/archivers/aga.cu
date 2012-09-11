#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "aga.h"
#include "../config.h"
#include "../utils.h"

inline FLOAT get_objective(struct solution *s, int objective) {
    if (objective == 0) {
        return s->makespan;
    } else if (objective == 1) {
        return s->energy_consumption;
    } else {
        assert(false);
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
        assert(n_pos < state->new_solutions_size);
        assert(s_pos < state->population_size);
        
        if (state->new_solutions[n_pos].initialized != SOLUTION__IN_USE) {
            n_pos++;
        } else {
            if (state->population[s_pos].initialized == SOLUTION__IN_USE) {
                s_pos++;
            } else {
                #ifdef DEBUG_3
                    fprintf(stderr, "[DEBUG] Found empty spot in archive %d!!!", s_pos);
                #endif
                
                if (state->population[s_pos].initialized == SOLUTION__NOT_INITIALIZED) {
                    create_empty_solution(&state->population[s_pos],
                        state->new_solutions[n_pos].s, 
                        state->new_solutions[n_pos].etc, 
                        state->new_solutions[n_pos].energy);
                }
                
                clone_solution(&state->population[s_pos], &state->new_solutions[n_pos]);
                
                // Marco la solución como ya creada e invalido la copia en la población de nuevas soluciones.
                state->new_solutions[n_pos].initialized = SOLUTION__EMPTY;
                new_solutions_count--;
                
                // Cambio la asignación del square de la solución
                state->grid_sol_loc[s_pos] = state->grid_sol_loc[n_pos + state->population_size];
                state->population_tag[s_pos] = state->new_solutions_tag[n_pos];
                state->tag_count[state->population_tag[s_pos]]++;
                state->population_count++; 
                                
                rc++;
            }
        }
    }

    n_pos = 0;

    while (new_solutions_count > 0)  // Some new solutions are non-dominated.
    {
        assert(n_pos < state->new_solutions_size);
        
        if (state->new_solutions[n_pos].initialized == SOLUTION__EMPTY) {
            n_pos++;
        } else {
            #ifdef DEBUG_3
                fprintf(stderr, "[DEBUG] Trying to add new solution %d to archive\n", n_pos);
            #endif
            
            int to_replace;
            to_replace = -1;

            int new_solution_grid_pos;
            new_solution_grid_pos = state->grid_sol_loc[n_pos + state->population_size];

            int most_square_pop;
            most_square_pop = state->grid_pop[new_solution_grid_pos];

            for (int i = 0; i < state->population_size; i++)
            {
                if (state->population[i].initialized == SOLUTION__IN_USE) {
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

            #ifdef DEBUG_3
                fprintf(stderr, "[DEBUG] > new solution %d is replacing solution %d\n", n_pos, to_replace);
            #endif

            if (to_replace != -1)
            {
                clone_solution(&state->population[to_replace], &state->new_solutions[n_pos]);
                state->tag_count[state->population_tag[to_replace]]--;
                state->population_tag[to_replace] = state->new_solutions_tag[n_pos];
                state->tag_count[state->new_solutions_tag[n_pos]]++;
                
                // Marco la solución como ya creada e invalido la copia en la población de nuevas soluciones.
                state->new_solutions[n_pos].initialized = SOLUTION__EMPTY;

                // Decremento la población en el square de la solución que acabo de borrar
                state->grid_pop[state->grid_sol_loc[to_replace]]--;
                
                // Cambio la asignación del square de la solución
                state->grid_sol_loc[to_replace] = state->grid_sol_loc[n_pos + state->population_size];

                //state->population_count++;
                rc++;
            }
            else
            {
                // Marco la solución como ya creada e invalido la copia en la población de nuevas soluciones.
                state->new_solutions[n_pos].initialized = SOLUTION__EMPTY;

                // Decremento la población en el square de la solución que acabo de borrar
                state->grid_pop[state->grid_sol_loc[n_pos + state->population_size]]--;
            }
            
            new_solutions_count--;
        }
    }

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
    FLOAT width[ARCHIVER__OBJECTIVES];

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

    FLOAT offset[ARCHIVER__OBJECTIVES];
    FLOAT largest[ARCHIVER__OBJECTIVES];

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
                if (state->population[a].initialized == SOLUTION__IN_USE) {
                    if (get_objective(&(state->population[a]), b) < offset[b]) {
                        offset[b] = get_objective(&(state->population[a]), b);
                    }

                    if (get_objective(&(state->population[a]), b) > largest[b]) {
                        largest[b] = get_objective(&(state->population[a]), b);
                    }
                }
            } else {
                if (state->new_solutions[a - state->population_size].initialized == SOLUTION__IN_USE) {
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

    int square;

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
            if (state->population[a].initialized == SOLUTION__IN_USE) {
                square = find_loc(state, a);
                state->grid_sol_loc[a] = square;
                state->grid_pop[square]++;
                                
                #ifdef DEBUG_3
                    fprintf(stderr, "[DEBUG] Population %d of the archive is in square %d. Now there are %d sols in square %d.\n",
                        a, square, state->grid_pop[square], square);
                #endif
            }
        } else {
            if (state->new_solutions[a - state->new_solutions_size].initialized == SOLUTION__IN_USE) {
                square = find_loc(state, a);
                state->grid_sol_loc[a] = square;
                state->grid_pop[square]++;
                
                #ifdef DEBUG_3
                    fprintf(stderr, "[DEBUG] Population %d of the new solutions is in square %d. Now there are %d sols in square %d.\n",
                        a - state->new_solutions_size, square, state->grid_pop[square], square);
                #endif
            }
        }
    }
}

void archivers_aga_init(struct aga_state *state, int population_max_size,
    struct solution *new_solutions, int *new_solutions_tag, int new_solutions_size,
    int tag_max) {
        
    state->population = (struct solution*)(malloc(sizeof(struct solution) * population_max_size));
    state->population_tag = (int*)(malloc(sizeof(int) * population_max_size));
    state->population_size = population_max_size;
    state->population_count = 0;

    for (int p = 0; p < state->population_size; p++) {
        state->population[p].initialized = SOLUTION__NOT_INITIALIZED;
        state->population_tag[p] = -1;
    }
    
    state->tag_max = tag_max;
    state->tag_count = (int*)malloc(sizeof(int) * tag_max);
    
    for (int p = 0; p < state->tag_max; p++) {
        state->tag_count[p] = 0;
    }

    state->new_solutions = new_solutions;
    state->new_solutions_tag = new_solutions_tag;
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
    for (int p = 0; p < state->population_size; p++) {
        if (state->population[p].initialized != SOLUTION__NOT_INITIALIZED) {
            free_solution(&state->population[p]);
        }
    }
    
    free(state->tag_count);
    free(state->population_tag);
    free(state->population);
    free(state->grid_pop);
    free(state->grid_sol_loc);
}

int delete_dominated_solutions(struct aga_state *state) {   
    // Elimino las soluciones dominadas entre si de la población de nuevas soluciones.
    for (int n_pos = 0; n_pos < state->new_solutions_size; n_pos++)
    {
        if (state->new_solutions[n_pos].initialized == SOLUTION__IN_USE) 
        {
            for (int n2_pos = n_pos+1; (n2_pos < state->new_solutions_size) && 
                (state->new_solutions[n_pos].initialized == SOLUTION__IN_USE); n2_pos++)
            {
                if (state->new_solutions[n2_pos].initialized == SOLUTION__IN_USE) 
                {
                    // Calculo no dominancia del elemento nuevo con el actual.
                    if ((state->new_solutions[n2_pos].makespan <= state->new_solutions[n_pos].makespan) && 
                        (state->new_solutions[n2_pos].energy_consumption <= state->new_solutions[n_pos].energy_consumption))
                    {                       
                        state->new_solutions[n_pos].initialized = SOLUTION__EMPTY;
                    }
                    else if ((state->new_solutions[n_pos].makespan <= state->new_solutions[n2_pos].makespan) && 
                        (state->new_solutions[n_pos].energy_consumption <= state->new_solutions[n2_pos].energy_consumption))
                    {
                        state->new_solutions[n2_pos].initialized = SOLUTION__EMPTY;
                    }
                }
            }
        }
    }
    
    int cant_new_solutions = 0;
    
    #ifdef DEBUG_3
        int current_archive_size = state->population_count;
    #endif    
    
    // Comparo la dominancia de las nuevas soluciones con las viejas soluciones.
    for (int n_pos = 0; n_pos < state->new_solutions_size; n_pos++)
    {
        if (state->new_solutions[n_pos].initialized == SOLUTION__IN_USE) 
        {
            for (int s_pos = 0; (s_pos < state->population_size) && 
                (state->new_solutions[n_pos].initialized == SOLUTION__IN_USE); s_pos++)
            {
                if (state->population[s_pos].initialized == SOLUTION__IN_USE) 
                {
                    // Calculo no dominancia del elemento nuevo con el actual.
                    if ((state->population[s_pos].makespan <= state->new_solutions[n_pos].makespan) && 
                        (state->population[s_pos].energy_consumption <= state->new_solutions[n_pos].energy_consumption))
                    {
                        // La nueva solucion es dominada por una ya existente.                        
                        state->new_solutions[n_pos].initialized = SOLUTION__EMPTY;
                    }
                    else if ((state->new_solutions[n_pos].makespan <= state->population[s_pos].makespan) && 
                        (state->new_solutions[n_pos].energy_consumption <= state->population[s_pos].energy_consumption))
                    {
                        // La nueva solucion domina a una ya existente.
                        state->population[s_pos].initialized = SOLUTION__EMPTY;
                        state->population_count--;
                        state->tag_count[state->population_tag[s_pos]]--;
                    }
                }
            }
        }
        
        if (state->new_solutions[n_pos].initialized == SOLUTION__IN_USE) 
        {
            cant_new_solutions++;
        }
    }
    
    #ifdef DEBUG_3
        fprintf(stderr, "[DEBUG] %d solutions were deleted form the archive\n", current_archive_size - state->population_count);
    #endif  
    
    return cant_new_solutions;
}

void archive_add_all_new(struct aga_state *state) {
    for (int n_pos = 0; n_pos < state->new_solutions_size; n_pos++)
    {
        if (state->new_solutions[n_pos].initialized == SOLUTION__IN_USE) {           
            for (int s_pos = 0; (s_pos < state->population_size) && 
                (state->new_solutions[n_pos].initialized == SOLUTION__IN_USE); s_pos++)
            {
                if (state->population[s_pos].initialized != SOLUTION__IN_USE) 
                {
                    #ifdef DEBUG_3
                        fprintf(stderr, "> found empty space in %d\n", s_pos);
                    #endif
                    
                    if (state->population[s_pos].initialized == SOLUTION__NOT_INITIALIZED) {
                        create_empty_solution(&state->population[s_pos],
                            state->new_solutions[n_pos].s, 
                            state->new_solutions[n_pos].etc, 
                            state->new_solutions[n_pos].energy);
                    }
                    
                    clone_solution(&state->population[s_pos], &state->new_solutions[n_pos]);
                    
                    state->population_tag[s_pos] = state->new_solutions_tag[n_pos];
                    state->tag_count[state->population_tag[s_pos]]++;
                    state->population_count++;
                    state->new_solutions[n_pos].initialized = SOLUTION__EMPTY;
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
    if (new_solutions + state->population_count <= state->population_size) {
        #ifdef DEBUG_3
            fprintf(stderr, "> there is room in the archive for all gathered solutions\n");
        #endif    
        archive_add_all_new(state);
    } else {
        #ifdef DEBUG_3
            fprintf(stderr, "> %d new solutions, but there's only space for %d solutions\n", 
                new_solutions, state->population_size - state->population_count);
        #endif   
        
        // Calculate grid location of mutant solution and renormalize archive if necessary.
        update_grid(state);

        // Update the archive by removing all dominated individuals.
        new_solutions = archive_add_solution(state, new_solutions);
    }
    
    #ifdef DEBUG_3
        fprintf(stderr, "[DEBUG] Current archive (%d):\n", state->population_count);
        for (int i = 0; i < state->population_size; i++) {
            if (state->population[i].initialized == SOLUTION__IN_USE) {
                fprintf(stderr, "(%d) %f %f %d\n",
                    i,
                    state->population[i].makespan, 
                    state->population[i].energy_consumption,
                    state->population_tag[i]);
            }
        }
    #endif
    
    return new_solutions;
}

void archivers_aga_dump(struct aga_state *state) {
    fprintf(stdout, "%d\n", state->population_count);
    for (int i = 0; i < state->population_size; i++) {
        if (state->population[i].initialized == SOLUTION__IN_USE) {
            for (int task_id = 0; task_id < state->population[i].etc->tasks_count; task_id++) {
                fprintf(stdout, "%d\n", state->population[i].task_assignment[task_id]);
            }
        }
    }
}

void archivers_aga_show(struct aga_state *state) {
    fprintf(stderr, "[DEBUG] ================================================\n");
    fprintf(stderr, "[DEBUG] Elite archive solutions [makespan energy origin]\n");
    fprintf(stderr, "[DEBUG] ================================================\n");

    for (int i = 0; i < state->population_size; i++) {
        if (state->population[i].initialized == SOLUTION__IN_USE) {
            fprintf(stderr, "%f %f %d\n",
                state->population[i].makespan, 
                state->population[i].energy_consumption,
                state->population_tag[i]);
        }
    }
    fprintf(stderr, "[DEBUG] ================================================\n");
    fprintf(stderr, "[DEBUG] Total solutions: %d\n", state->population_count);
    fprintf(stderr, "[DEBUG] ================================================\n");
}
