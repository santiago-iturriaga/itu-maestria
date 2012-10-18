#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>

#include "aga.h"
#include "../random/cpu_mt.h"

struct aga_state AGA;

int find_loc(int solution_pos);
void update_grid(int new_solution_pos);

void archivers_aga_init();
int archivers_aga_add(int new_solution_pos);
void archivers_aga_free();

void archivers_aga() {
    archivers_aga_init();

    MPI_Status status;
    int finalize = 0, aux, msg_count;
    int rc;
    
    #ifndef NDEBUG
        int iteration = 0;
    #endif
    
    struct solution input_buffer[ARCHIVER__MAX_INPUT_BUFFER];

    while (finalize == 0) {
        #ifndef NDEBUG
            iteration++;
            fprintf(stderr, "[DEBUG][AGA] ITERATION %d ==================================\n", iteration);
            fprintf(stderr, "[DEBUG][AGA] Current population = %d\n", AGA.population_count);
            fprintf(stderr, "[DEBUG][AGA] Waiting for a message.\n");
        #endif
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == AGA__NEW_SOL_MSG) {
            MPI_Get_count(&status, mpi_solution_type, &msg_count);
            
            #ifndef NDEBUG
                fprintf(stderr, "[DEBUG][AGA] New solution received (count=%d)\n", msg_count);
            #endif
            
            MPI_Recv(input_buffer, msg_count, mpi_solution_type, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            
            for (int s = 0; s < msg_count; s++) {
                input_buffer[s].status = SOLUTION__STATUS_NEW;

                #ifndef NDEBUG
                    fprintf(stderr, "[DEBUG][AGA] Received (energy=%.2f / coverage=%.2f / nforwardings=%.2f)\n", 
                        input_buffer[s].energy, input_buffer[s].coverage, input_buffer[s].nforwardings);
                #endif

                int sol_idx = -1;
                for (int i = 0; (i < AGA__MAX_ARCHIVE_SIZE) && (sol_idx < AGA.population_count) && (input_buffer[s].status == SOLUTION__STATUS_NEW); i++) {
                    if (AGA.population[i].status == SOLUTION__STATUS_EMPTY) {
                        // Encontré una posición libre. Agrego la solución al archivo acá.
                        clone_solution(&AGA.population[i], &input_buffer[s]);
                        AGA.population[i].coverage = 1/AGA.population[i].coverage;
                        
                        rc = archivers_aga_add(i);
                        
                        #ifndef NDEBUG
                            if (rc > 0) {
                                fprintf(stderr, "[DEBUG][AGA] Solution in %d was added.\n", i);
                            } else {
                                fprintf(stderr, "[DEBUG][AGA] Solution in %d was discarded.\n", i);
                            }
                        #endif
                        
                        input_buffer[s].status = SOLUTION__STATUS_EMPTY;                    
                        
                    } else if (AGA.population[i].status == SOLUTION__STATUS_READY) {
                        // Posición ocupada, sigo buscando.
                        sol_idx++;
                    }
                }

                assert(input_buffer[s].status == SOLUTION__STATUS_EMPTY);
            }
            
            assert(AGA.population_count > 0);
        } else if (status.MPI_TAG == AGA__EXIT_MSG) {
            MPI_Get_count(&status, mpi_solution_type, &msg_count);
            
            #ifndef NDEBUG
                fprintf(stderr, "[DEBUG][AGA] Terminate message received\n");
            #endif
            MPI_Recv(&aux, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            fprintf(stderr, "[INFO][AGA] Kaput!\n");
            finalize = 1;
            
        } else {
            fprintf(stderr, "[ERROR][AGA] Unknown TAG %d\n", status.MPI_TAG);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
    }

    fprintf(stderr, "[INFO] AGA information ============================================ \n");
    fprintf(stderr, "[INFO] Population count: %d\n", AGA.population_count);
    fprintf(stderr, "[INFO] Population:\n");
    
    fprintf(stdout, "\nid,borders_threshold,margin_forwarding,min_delay,max_delay,neighbors_threshold,energy,coverage,nforwardings,time\n");
    
    for (int i = 0; i < AGA__MAX_ARCHIVE_SIZE; i++) {
        if (AGA.population[i].status == SOLUTION__STATUS_READY) {
            fprintf(stderr, "(%d) borders_threshold   = %.2f\n", i, AGA.population[i].borders_threshold);
            fprintf(stderr, "     margin_forwarding   = %.2f\n", AGA.population[i].margin_forwarding);
            fprintf(stderr, "     min_delay           = %.2f\n", AGA.population[i].min_delay);
            fprintf(stderr, "     max_delay           = %.2f\n", AGA.population[i].max_delay);
            fprintf(stderr, "     neighbors_threshold = %d\n", AGA.population[i].neighbors_threshold);
            fprintf(stderr, "     energy              = %.2f\n", AGA.population[i].energy);
            fprintf(stderr, "     coverage            = %.2f (%.2f)\n", 1/AGA.population[i].coverage, AGA.population[i].coverage);
            fprintf(stderr, "     nforwardings        = %.2f\n", AGA.population[i].nforwardings);
            fprintf(stderr, "     time                = %.2f\n", AGA.population[i].time);

            fprintf(stderr, "%d,%f,%f,%f,%f,%d,%f,%f,%f,%f\n", i,
                AGA.population[i].borders_threshold, AGA.population[i].margin_forwarding,
                AGA.population[i].min_delay, AGA.population[i].max_delay, 
                AGA.population[i].neighbors_threshold,
                AGA.population[i].energy, 1/AGA.population[i].coverage,
                AGA.population[i].nforwardings, AGA.population[i].time);
            
            fprintf(stdout, "%d,%f,%f,%f,%f,%d,%f,%f,%f,%f\n", i,
                AGA.population[i].borders_threshold, AGA.population[i].margin_forwarding,
                AGA.population[i].min_delay, AGA.population[i].max_delay, 
                AGA.population[i].neighbors_threshold,
                AGA.population[i].energy, 1/AGA.population[i].coverage,
                AGA.population[i].nforwardings, AGA.population[i].time);
        }
    }

    archivers_aga_free();
}

int archivers_aga_add(int new_solution_pos)
{
    // Calculate grid location of mutant solution and renormalize archive if necessary.
    update_grid(new_solution_pos);

    // Update the archive by removing all dominated individuals.
    // Given a solution s (new_solution_pos), add it to the archive if:
    //   a) the archive is empty
    //   b) the archive is not full and s is not dominated or equal to anything currently in the archive
    //   c) s dominates anything in the archive
    //   d) the archive is full but s is nondominated and is in a no more crowded square than at least one solution
    //      in addition, maintain the archive such that all solutions are nondominated.

    double obj_current[OBJECTIVES];
    double obj_new[OBJECTIVES];

    for (int i = 0; i < OBJECTIVES; i++)
    {
        obj_new[i] = get_objective(&(AGA.population[new_solution_pos]), i);
    }

    int solutions_deleted = 0;
    int new_solution_is_dominated = 0;

    int aux_dominates, aux_is_dominated;

    int s_idx = -1;
    for (int s_pos = 0; (s_pos < AGA__MAX_ARCHIVE_SIZE) && (s_idx < AGA.population_count) && (new_solution_is_dominated == 0); s_pos++)
    {
        if ((AGA.population[s_pos].status > SOLUTION__STATUS_EMPTY) && (s_pos != new_solution_pos))
        {
            s_idx++;
            
            aux_dominates = 1;
            aux_is_dominated = 1;

            // Calculo no dominancia del elemento nuevo con el actual.
            for (int i = 0; i < OBJECTIVES; i++)
            {
                obj_current[i] = get_objective(&(AGA.population[s_pos]), i);

                if (obj_current[i] > obj_new[i]) {
                    aux_is_dominated = 0;
                }

                if (obj_current[i] < obj_new[i]) {
                    aux_dominates = 0;
                }
            }

            if (aux_is_dominated == 1)
            {
                // La nueva solucion es dominada por una ya existente.
                new_solution_is_dominated = 1;
                
                #ifndef NDEBUG
                    fprintf(stderr, "[DEBUG] New solution is dominated by %d\n", s_pos);
                #endif
            }
            else if (aux_dominates == 1)
            {
                // La nueva solucion domina a una ya existente.
                solutions_deleted++;
                AGA.population[s_pos].status = SOLUTION__STATUS_EMPTY;
                
                #ifndef NDEBUG
                    fprintf(stderr, "[DEBUG] New solution dominates %d\n", s_pos);
                #endif
            }
        }
    }

    if (new_solution_is_dominated == 0)  // Solution is non-dominated by the list.
    {
        AGA.population_count -= solutions_deleted;

        if (AGA.population_count + 1 == AGA__MAX_ARCHIVE_SIZE)
        {
            // El archivo esta lleno. Tengo que reemplazar alguna solución.
            int to_replace = new_solution_pos;

            int new_solution_grid_pos;
            new_solution_grid_pos = AGA.grid_sol_loc[new_solution_pos];

            int most_square_pop;
            most_square_pop = AGA.grid_pop[new_solution_grid_pos];

            for (int i = 0; i < AGA__MAX_ARCHIVE_SIZE; i++)
            {
                if ((AGA.population[i].status > SOLUTION__STATUS_EMPTY) && (i != new_solution_pos)) {
                    int i_grid_pos;
                    i_grid_pos = AGA.grid_sol_loc[i];

                    int i_square_pop;
                    i_square_pop = AGA.grid_pop[i_grid_pos];

                    if (i_square_pop > most_square_pop)
                    {
                        most_square_pop = AGA.grid_pop[i_square_pop];
                        to_replace = i;
                    }
                }
            }

            if (to_replace != new_solution_pos)
            {
                AGA.population[to_replace].status = SOLUTION__STATUS_EMPTY;
                AGA.population[new_solution_pos].status = SOLUTION__STATUS_READY;

                return 1;
            }
            else
            {
                AGA.population[new_solution_pos].status = SOLUTION__STATUS_EMPTY;

                return -1;
            }
        }
        else
        {
            AGA.population[new_solution_pos].status = SOLUTION__STATUS_READY;
            AGA.population_count++;

            return 1;
        }
    } else {
        AGA.population[new_solution_pos].status = SOLUTION__STATUS_EMPTY;

        return -1;
    }
}

void archivers_aga_init() {
    AGA.max_locations = pow(2, ARCHIVER__AGA_DEPTH * OBJECTIVES); // Number of locations in grid.
    AGA.grid_pop = (int*)(malloc(sizeof(int) * (AGA.max_locations + 1)));

    for (int i = 0; i < OBJECTIVES; i++) {
        AGA.gl_offset[i] = 0.0;
        AGA.gl_range[i] = 0.0;
        AGA.gl_largest[i] = 0.0;
    }

    for (int i = 0; i < AGA.max_locations; i++) {
        AGA.grid_pop[i] = 0;
    }
    
    for (int i = 0; i < AGA__MAX_ARCHIVE_SIZE; i++) {
        AGA.population[i].status = SOLUTION__STATUS_EMPTY;
    }
}

void archivers_aga_free() {
    free(AGA.grid_pop);
}

inline int find_loc(int solution_pos)
{
    // Find the grid location of a solution given a vector of its objective values.
    int loc = 0;

    int d;
    int n = 1;

    int i;

    int inc[OBJECTIVES];
    double width[OBJECTIVES];

    // if the solution is out of range on any objective, return 1 more than the maximum possible grid location number
    for (i = 0; i < OBJECTIVES; i++)
    {
        if ((get_objective(&(AGA.population[solution_pos]), i) < AGA.gl_offset[i]) ||
            (get_objective(&(AGA.population[solution_pos]), i) > AGA.gl_offset[i] + AGA.gl_range[i]))

            return AGA.max_locations;
    }

    for (i = 0; i < OBJECTIVES; i++)
    {
        inc[i] = n;
        n *=2;
        width[i] = AGA.gl_range[i];
    }

    for (d = 1; d <= ARCHIVER__AGA_DEPTH; d++)
    {
        for (i = 0; i < OBJECTIVES; i++)
        {
            if(get_objective(&(AGA.population[solution_pos]), i) < width[i]/2 + AGA.gl_offset[i])
                loc += inc[i];
            else
                AGA.gl_offset[i] += width[i]/2;
        }
        for (i = 0; i < OBJECTIVES; i++)
        {
            inc[i] *= (OBJECTIVES *2);
            width[i] /= 2;
        }
    }

    return loc;
}

inline void update_grid(int new_solution_pos)
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
        for (a = 0; a < AGA__MAX_ARCHIVE_SIZE; a++)
        {
            if ((AGA.population[a].status > SOLUTION__STATUS_EMPTY) || (a == new_solution_pos)) {
                if (get_objective(&(AGA.population[a]), b) < offset[b])
                    offset[b] = get_objective(&(AGA.population[a]), b);

                if (get_objective(&(AGA.population[a]), b) > largest[b])
                    largest[b] = get_objective(&(AGA.population[a]), b);
            }
        }
    }

    double sse = 0;
    double product = 1;

    for (a = 0; a < OBJECTIVES; a++)
    {
        sse += ((AGA.gl_offset[a] - offset[a]) * (AGA.gl_offset[a] - offset[a]));
        sse += ((AGA.gl_largest[a] - largest[a]) * (AGA.gl_largest[a] - largest[a]));
        product *= AGA.gl_range[a];
    }

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
            AGA.gl_largest[a] = largest[a] + 0.2 * largest[a];
            AGA.gl_offset[a] = offset[a] + 0.2 * offset[a];
            AGA.gl_range[a] = AGA.gl_largest[a] - AGA.gl_offset[a];
        }

        for (a = 0; a < AGA.max_locations; a++)
        {
            AGA.grid_pop[a] = 0;
        }

        for (a = 0; a < AGA__MAX_ARCHIVE_SIZE; a++)
        {
            if ((AGA.population[a].status > SOLUTION__STATUS_EMPTY) || (a == new_solution_pos)) {
                square = find_loc(a);

                AGA.grid_sol_loc[a] = square;
                AGA.grid_pop[square]++;
            }
        }
    } else {
        square = find_loc(new_solution_pos);

        AGA.grid_sol_loc[new_solution_pos] = square;
        AGA.grid_pop[square]++;
    }
}
