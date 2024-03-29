#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "aga.h"
#include "../../random/cpu_mt.h"

int archive_add_solution(struct mls_thread_arg *instance, int new_solution_pos)
{   
    // Given a solution s (new_solution_pos), add it to the archive if:
    //   a) the archive is empty
    //   b) the archive is not full and s is not dominated or equal to anything currently in the archive
    //   c) s dominates anything in the archive
    //   d) the archive is full but s is nondominated and is in a no more crowded square than at least one solution
    //      in addition, maintain the archive such that all solutions are nondominated.

    float makespan_new, energy_new;
    makespan_new = get_objective(&(instance->population[new_solution_pos]), SOLUTION__MAKESPAN_OBJ);
    energy_new = get_objective(&(instance->population[new_solution_pos]), SOLUTION__ENERGY_OBJ);

    int solutions_deleted = 0;
    int new_solution_is_dominated = 0;

    int s_idx = -1;
    for (int s_pos = 0; s_pos < instance->population_max_size; s_pos++)
    {
        if ((instance->population[s_pos].status > SOLUTION__STATUS_EMPTY) &&
            (s_pos != new_solution_pos))
        {
            s_idx++;

            // Calculo no dominancia del elemento nuevo con el actual.
            float makespan, energy;
            makespan = get_objective(&(instance->population[s_pos]), SOLUTION__MAKESPAN_OBJ);
            energy = get_objective(&(instance->population[s_pos]), SOLUTION__ENERGY_OBJ);

            #if defined(DEBUG_DEV) 
            fprintf(stdout, "[%d] Makespan: %f %f || Energy %f %f\n", 
                s_pos, makespan, makespan_new, energy, energy_new);
            #endif

            if ((makespan <= makespan_new) && (energy <= energy_new))
            {
                // La nueva solucion es dominada por una ya existente.
                new_solution_is_dominated = 1;

                #if defined(DEBUG_DEV) 
                fprintf(stdout, "[DEBUG] Individual %d[%f,%f] is dominated by %d[%f,%f]\n", 
                    new_solution_pos, makespan_new, energy_new, s_pos, makespan, energy);
                #endif
            }
            else if ((makespan_new <= makespan) && (energy_new <= energy))
            {
                // La nueva solucion domina a una ya existente.
                solutions_deleted++;
                instance->population_count[0] = instance->population_count[0] - 1;
                instance->population[s_pos].status = SOLUTION__STATUS_EMPTY;

                #if defined(DEBUG_DEV) 
                fprintf(stdout, "[DEBUG] Removed individual %d[%f,%f] because %d[%f,%f] is better\n", 
                    s_pos, makespan, energy, new_solution_pos, makespan_new, energy_new);
                #endif
            }
        } 
        else if (instance->population[s_pos].status == SOLUTION__STATUS_TO_DEL) 
        {
            // Calculo no dominancia del elemento nuevo con el actual.
            float makespan, energy;
            makespan = get_objective(&(instance->population[s_pos]), SOLUTION__MAKESPAN_OBJ);
            energy = get_objective(&(instance->population[s_pos]), SOLUTION__ENERGY_OBJ);
            
            if ((makespan_new <= makespan) && (energy_new <= energy))
            {
                // La nueva solucion domina a una ya existente que esta para borrar. La borro del todo.
                instance->population[s_pos].status = SOLUTION__STATUS_EMPTY;

                #if defined(DEBUG_DEV) 
                fprintf(stdout, "[DEBUG] Removed TO_DEL individual %d because %d is better\n", s_pos, new_solution_pos);
                #endif
            }
        }
    }

    if (new_solution_is_dominated == 0)  // Solution is non-dominated by the list.
    {            
        if ((instance->population_count[0] + instance->count_threads) >= instance->population_max_size)
        {
            int to_replace = new_solution_pos;

            int new_solution_grid_pos;
            new_solution_grid_pos = instance->archiver_state->grid_sol_loc[new_solution_pos];
                                           
            int most_square_pop;
            most_square_pop = instance->archiver_state->grid_pop[new_solution_grid_pos];
         
            for (int i = 0; i < instance->population_max_size; i++)
            {
                if ((instance->population[i].status == SOLUTION__STATUS_READY) && 
                    (i != new_solution_pos)) {
                        
                    int i_grid_pos;
                    i_grid_pos = instance->archiver_state->grid_sol_loc[i];
                                
                    int i_square_pop;
                    i_square_pop = instance->archiver_state->grid_pop[i_grid_pos];

                    if (i_square_pop > most_square_pop)
                    {
                        most_square_pop = instance->archiver_state->grid_pop[i_square_pop];
                        to_replace = i;
                    }
                }
            }
            
            if (to_replace != new_solution_pos) 
            {
                instance->population[to_replace].status = SOLUTION__STATUS_TO_DEL;
                instance->population[new_solution_pos].status = SOLUTION__STATUS_READY;
                        
                return 1;
            }
            else 
            {
                instance->population[new_solution_pos].status = SOLUTION__STATUS_TO_DEL;
                
                return -1;
            }
        }
        else
        {
            instance->population[new_solution_pos].status = SOLUTION__STATUS_READY;
            instance->population_count[0] = instance->population_count[0] + 1;

            #if defined(DEBUG_DEV) 
            fprintf(stdout, "[DEBUG] Added invidiual %d because is ND\n", new_solution_pos);
            #endif
            
            return 1;
        }
    } else {
        instance->population[new_solution_pos].status = SOLUTION__STATUS_EMPTY;
        
        return -1;
    }
}

int find_loc(struct mls_thread_arg *instance, int solution_pos)
{
    // Find the grid location of a solution given a vector of its objective values.
    int loc = 0;
    
    int d;
    int n = 1;

    int i;

    int inc[OBJECTIVES];
    double width[OBJECTIVES];

    // printf("obj = %d, depth = %d\n", objectives, depth);

    // if the solution is out of range on any objective, return 1 more than the maximum possible grid location number
    for (i = 0; i < OBJECTIVES; i++)
    {
        if ((get_objective(&(instance->population[solution_pos]), i) < instance->archiver_state->gl_offset[i]) ||
            (get_objective(&(instance->population[solution_pos]), i) > instance->archiver_state->gl_offset[i] + instance->archiver_state->gl_range[i]))
            
            return instance->archiver_state->max_locations;
    }

    for (i = 0; i < OBJECTIVES; i++)
    {
        inc[i] = n;
        n *=2;
        width[i] = instance->archiver_state->gl_range[i];
    }

    for (d = 1; d <= ARCHIVER__AGA_DEPTH; d++)
    {
        for (i = 0; i < OBJECTIVES; i++)
        {
            if(get_objective(&(instance->population[solution_pos]), i) < width[i]/2 + instance->archiver_state->gl_offset[i])
                loc += inc[i];
            else
                instance->archiver_state->gl_offset[i] += width[i]/2;
        }
        for (i = 0; i < OBJECTIVES; i++)
        {
            inc[i] *= (OBJECTIVES *2);
            width[i] /= 2;
        }
    }
    
    return loc;
}

void update_grid(struct mls_thread_arg *instance, int new_solution_pos)
{
    // recalculate ranges for grid in the light of a new solution s
    int a, b;
    
    double offset[OBJECTIVES];
    double largest[OBJECTIVES];

    for (a = 0; a < OBJECTIVES; a++)
    {
        offset[a] = ARCHIVER__AGA_LARGE;
        largest[a] = -ARCHIVER__AGA_LARGE;
    }

    for (b = 0; b < OBJECTIVES; b++)
    {
        for (a = 0; a < instance->population_max_size; a++)
        {
            if ((instance->population[a].status > SOLUTION__STATUS_EMPTY) || (a == new_solution_pos)) {           
                if (get_objective(&(instance->population[a]), b) < offset[b])
                    offset[b] = get_objective(&(instance->population[a]), b);
                    
                if (get_objective(&(instance->population[a]), b) > largest[b])
                    largest[b] = get_objective(&(instance->population[a]), b);
            }
        }
    }
    // printf("oldCURENT:largest = %f, offset = %f\n", largest[0], offset[0]);
    // printf("oldCURENT:largest = %f, offset = %f\n", largest[1], offset[1]);

    double sse = 0;
    double product = 1;

    for (a = 0; a < OBJECTIVES; a++)
    {
        sse += ((instance->archiver_state->gl_offset[a] - offset[a]) * (instance->archiver_state->gl_offset[a] - offset[a]));
        sse += ((instance->archiver_state->gl_largest[a] - largest[a]) * (instance->archiver_state->gl_largest[a] - largest[a]));
        product *= instance->archiver_state->gl_range[a];
    }

    // printf("sse = %f\n", sse);

    static int change = 0;
    int square;
   
    // If the summed squared error (difference) between old and new
    // minima and maxima in each of the objectives
    // is bigger than 10 percent of the square of the size of the space
    // then renormalise the space and recalculte grid locations
    if (sse > (0.1 * product * product))
    {                            
        change++;                

        for (a = 0; a < OBJECTIVES; a++)
        {
            instance->archiver_state->gl_largest[a] = largest[a] + 0.2 * largest[a];
            instance->archiver_state->gl_offset[a] = offset[a] + 0.2 * offset[a];
            instance->archiver_state->gl_range[a] = instance->archiver_state->gl_largest[a] 
                - instance->archiver_state->gl_offset[a];
        }

        for (a = 0; a < instance->archiver_state->max_locations; a++)
        {
            instance->archiver_state->grid_pop[a] = 0;
        }

        for (a = 0; a < instance->population_max_size; a++)
        {
            if ((instance->population[a].status > SOLUTION__STATUS_EMPTY) ||
                (a == new_solution_pos)) {
                
                square = find_loc(instance, a);
                
                instance->archiver_state->grid_sol_loc[a] = square;
                instance->archiver_state->grid_pop[square]++;
            }
        }
    } else {
        square = find_loc(instance, new_solution_pos);
        
        instance->archiver_state->grid_sol_loc[new_solution_pos] = square;
        instance->archiver_state->grid_pop[square]++;
    }
}

void archivers_aga_init(struct mls_instance *instance) {
    instance->archiver_state->max_locations = pow(2, ARCHIVER__AGA_DEPTH * OBJECTIVES); // Number of locations in grid.
    instance->archiver_state->grid_pop = (int*)(malloc(sizeof(int) * (instance->archiver_state->max_locations + 1)));
    instance->archiver_state->grid_sol_loc = (int*)(malloc(sizeof(int) * instance->population_max_size));
    
    //fprintf(stdout, "starting grid_pop size: %d\n", instance->archiver_state->max_locations+1);
    
    for (int i = 0; i < OBJECTIVES; i++) {
        instance->archiver_state->gl_offset[i] = 0.0;
        instance->archiver_state->gl_range[i] = 0.0;
        instance->archiver_state->gl_largest[i] = 0.0;
    }
    
    for (int i = 0; i < instance->archiver_state->max_locations; i++) {
        instance->archiver_state->grid_pop[i] = 0;
    }
}

void archivers_aga_free(struct mls_instance *instance) {
    free(instance->archiver_state->grid_pop);
    free(instance->archiver_state->grid_sol_loc);
}

int archivers_aga(struct mls_thread_arg *instance, int new_solution_pos)
{   
    // Calculate grid location of mutant solution and renormalize archive if necessary.
    update_grid(instance, new_solution_pos);
    
    // Update the archive by removing all dominated individuals.
    int result = archive_add_solution(instance, new_solution_pos); 

    assert(instance->population[new_solution_pos].status != SOLUTION__STATUS_NOT_READY);
    assert(instance->population_count[0] > 0);

    return result;
}
