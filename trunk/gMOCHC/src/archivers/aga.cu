#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "aga.h"
#include "../config.h"
#include "../utils.h"

struct aga_state ARCHIVER;

inline FLOAT get_objective(struct solution *s, int objective) {
    if (objective == 0) {
        return s->makespan;
    } else if (objective == 1) {
        return s->energy_consumption;
    }

    assert(false);
    return 0;
}

int archive_add_solution(int new_solutions_count)
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

    while ((new_solutions_count > 0)&&(ARCHIVER.population_count < ARCHIVER__MAX_SIZE)) {
        assert(s_pos < ARCHIVER__MAX_SIZE);

        if (ARCHIVER.new_solutions[n_pos].initialized != SOLUTION__IN_USE) {
            n_pos++;
        } else {
            if (ARCHIVER.population[s_pos].initialized == SOLUTION__IN_USE) {
                s_pos++;
            } else {
                if (ARCHIVER.population[s_pos].initialized == SOLUTION__NOT_INITIALIZED) {
                    create_empty_solution(&ARCHIVER.population[s_pos]);
                }

                clone_solution(&ARCHIVER.population[s_pos], &ARCHIVER.new_solutions[n_pos]);

                // Marco la solución como ya creada e invalido la copia en la población de nuevas soluciones.
                ARCHIVER.new_solutions[n_pos].initialized = SOLUTION__EMPTY;
                new_solutions_count--;

                // Cambio la asignación del square de la solución
                ARCHIVER.grid_sol_loc[s_pos] = ARCHIVER.grid_sol_loc[n_pos + ARCHIVER__MAX_SIZE];
                ARCHIVER.population_tag[s_pos] = ARCHIVER.new_solutions_tag[n_pos];
                ARCHIVER.tag_count[ARCHIVER.population_tag[s_pos]]++;
                ARCHIVER.population_count++;

                rc++;
            }
        }
    }

    n_pos = 0;

    while (new_solutions_count > 0)  // Some new solutions are non-dominated.
    {
        if (ARCHIVER.new_solutions[n_pos].initialized == SOLUTION__EMPTY) {
            n_pos++;
        } else {
            int to_replace;
            to_replace = -1;

            int new_solution_grid_pos;
            new_solution_grid_pos = ARCHIVER.grid_sol_loc[n_pos + ARCHIVER__MAX_SIZE];

            int most_square_pop;
            most_square_pop = ARCHIVER.grid_pop[new_solution_grid_pos];

            for (int i = 0; i < ARCHIVER__MAX_SIZE; i++)
            {
                if (ARCHIVER.population[i].initialized == SOLUTION__IN_USE) {
                    int i_grid_pos;
                    i_grid_pos = ARCHIVER.grid_sol_loc[i];

                    int i_square_pop;
                    i_square_pop = ARCHIVER.grid_pop[i_grid_pos];

                    if (i_square_pop > most_square_pop)
                    {
                        most_square_pop = ARCHIVER.grid_pop[i_square_pop];
                        to_replace = i;
                    }
                }
            }

            if (to_replace != -1)
            {
                clone_solution(&ARCHIVER.population[to_replace], &ARCHIVER.new_solutions[n_pos]);
                ARCHIVER.tag_count[ARCHIVER.population_tag[to_replace]]--;
                ARCHIVER.population_tag[to_replace] = ARCHIVER.new_solutions_tag[n_pos];
                ARCHIVER.tag_count[ARCHIVER.new_solutions_tag[n_pos]]++;

                // Marco la solución como ya creada e invalido la copia en la población de nuevas soluciones.
                ARCHIVER.new_solutions[n_pos].initialized = SOLUTION__EMPTY;

                // Decremento la población en el square de la solución que acabo de borrar
                ARCHIVER.grid_pop[ARCHIVER.grid_sol_loc[to_replace]]--;

                // Cambio la asignación del square de la solución
                ARCHIVER.grid_sol_loc[to_replace] = ARCHIVER.grid_sol_loc[n_pos + ARCHIVER__MAX_SIZE];

                //state->population_count++;
                rc++;
            }
            else
            {
                // Marco la solución como ya creada e invalido la copia en la población de nuevas soluciones.
                ARCHIVER.new_solutions[n_pos].initialized = SOLUTION__EMPTY;

                // Decremento la población en el square de la solución que acabo de borrar
                ARCHIVER.grid_pop[ARCHIVER.grid_sol_loc[n_pos + ARCHIVER__MAX_SIZE]]--;
            }

            new_solutions_count--;
        }
    }

    return rc;
}

inline int find_loc(int solution_pos)
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
        if (solution_pos < ARCHIVER__MAX_SIZE) {
            if ((get_objective(&(ARCHIVER.population[solution_pos]), i) < ARCHIVER.gl_offset[i]) ||
                (get_objective(&(ARCHIVER.population[solution_pos]), i) > ARCHIVER.gl_offset[i] + ARCHIVER.gl_range[i])) {

                return ARCHIVER.max_locations;
            }
        } else {
            if ((get_objective(&(ARCHIVER.new_solutions[solution_pos - ARCHIVER__MAX_SIZE]), i) < ARCHIVER.gl_offset[i]) ||
                (get_objective(&(ARCHIVER.new_solutions[solution_pos - ARCHIVER__MAX_SIZE]), i) > ARCHIVER.gl_offset[i] + ARCHIVER.gl_range[i])) {

                return ARCHIVER.max_locations;
            }
        }
    }

    for (i = 0; i < ARCHIVER__OBJECTIVES; i++)
    {
        inc[i] = n;
        n *=2;
        width[i] = ARCHIVER.gl_range[i];
    }

    for (d = 1; d <= ARCHIVER__AGA_DEPTH; d++)
    {
        for (i = 0; i < ARCHIVER__OBJECTIVES; i++)
        {
            if (solution_pos < ARCHIVER__MAX_SIZE) {
                if(get_objective(&(ARCHIVER.population[solution_pos]), i) <
                    width[i]/2 + ARCHIVER.gl_offset[i]) {

                    loc += inc[i];
                } else {
                    ARCHIVER.gl_offset[i] += width[i]/2;
                }
            } else {
                if(get_objective(&(ARCHIVER.new_solutions[solution_pos - ARCHIVER__MAX_SIZE]), i) <
                    width[i]/2 + ARCHIVER.gl_offset[i]) {

                    loc += inc[i];
                } else {
                    ARCHIVER.gl_offset[i] += width[i]/2;
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

void update_grid(int new_solutions_size)
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
        for (a = 0; a < ARCHIVER__MAX_SIZE + new_solutions_size; a++)
        {
            if (a < ARCHIVER__MAX_SIZE) {
                if (ARCHIVER.population[a].initialized == SOLUTION__IN_USE) {
                    if (get_objective(&(ARCHIVER.population[a]), b) < offset[b]) {
                        offset[b] = get_objective(&(ARCHIVER.population[a]), b);
                    }

                    if (get_objective(&(ARCHIVER.population[a]), b) > largest[b]) {
                        largest[b] = get_objective(&(ARCHIVER.population[a]), b);
                    }
                }
            } else {
                if (ARCHIVER.new_solutions[a - ARCHIVER__MAX_SIZE].initialized == SOLUTION__IN_USE) {
                    if (get_objective(&(ARCHIVER.new_solutions[a - ARCHIVER__MAX_SIZE]), b) < offset[b]) {
                        offset[b] = get_objective(&(ARCHIVER.new_solutions[a - ARCHIVER__MAX_SIZE]), b);
                    }

                    if (get_objective(&(ARCHIVER.new_solutions[a - ARCHIVER__MAX_SIZE]), b) > largest[b]) {
                        largest[b] = get_objective(&(ARCHIVER.new_solutions[a - ARCHIVER__MAX_SIZE]), b);
                    }
                }
            }
        }
    }

    int square;

    for (a = 0; a < ARCHIVER__OBJECTIVES; a++)
    {
        ARCHIVER.gl_largest[a] = largest[a] + 0.2 * largest[a];
        ARCHIVER.gl_offset[a] = offset[a] + 0.2 * offset[a];
        ARCHIVER.gl_range[a] = ARCHIVER.gl_largest[a] - ARCHIVER.gl_offset[a];
    }

    for (a = 0; a < ARCHIVER.max_locations; a++)
    {
        ARCHIVER.grid_pop[a] = 0;
    }

    for (a = 0; a < ARCHIVER__MAX_SIZE + new_solutions_size; a++)
    {
        if (a < ARCHIVER__MAX_SIZE) {
            if (ARCHIVER.population[a].initialized == SOLUTION__IN_USE) {
                square = find_loc(a);
                ARCHIVER.grid_sol_loc[a] = square;
                ARCHIVER.grid_pop[square]++;
            }
        } else {
            if (ARCHIVER.new_solutions[a - new_solutions_size].initialized == SOLUTION__IN_USE) {
                square = find_loc(a);
                ARCHIVER.grid_sol_loc[a] = square;
                ARCHIVER.grid_pop[square]++;
            }
        }
    }
}

void archivers_aga_init(int tag_size) {
    fprintf(stderr, "[INFO] == AGA archiver configuration constants ==============\n");
    fprintf(stderr, "       ARCHIVER__MAX_SIZE                          : %d\n", ARCHIVER__MAX_SIZE);
    fprintf(stderr, "       ARCHIVER__OBJECTIVES                        : %d\n", ARCHIVER__OBJECTIVES);
    fprintf(stderr, "       ARCHIVER__AGA_DEPTH                         : %d\n", ARCHIVER__AGA_DEPTH);
    fprintf(stderr, "       ARCHIVER__MAX_NEW_SOLS                      : %d\n", ARCHIVER__MAX_NEW_SOLS);
    fprintf(stderr, "       ARCHIVER__MAX_TAGS                          : %d\n", ARCHIVER__MAX_TAGS);
    fprintf(stderr, "       tag_size                                    : %d\n", tag_size);
    fprintf(stderr, "[INFO] == AGA archiver configuration constants ==============\n");

    assert(tag_size <= ARCHIVER__MAX_TAGS);

    ARCHIVER.population_count = 0;

    for (int p = 0; p < ARCHIVER__MAX_SIZE; p++) {
        ARCHIVER.population[p].initialized = SOLUTION__NOT_INITIALIZED;
        ARCHIVER.population_tag[p] = -1;
    }

    ARCHIVER.tag_size = tag_size;

    for (int p = 0; p < ARCHIVER.tag_size; p++) {
        ARCHIVER.tag_count[p] = 0;
    }

    /* Estado de la población de intercambio */
    for (int i = 0; i < ARCHIVER__MAX_NEW_SOLS; i++) {
        create_empty_solution(&ARCHIVER.new_solutions[i]);
        ARCHIVER.new_solutions_tag[i] = 0;
    }

    ARCHIVER.max_locations = (int)(pow(2, ARCHIVER__AGA_DEPTH * ARCHIVER__OBJECTIVES)); // Number of locations in grid.
    ARCHIVER.grid_pop = (int*)(malloc(sizeof(int) * (ARCHIVER.max_locations + 1)));
    ARCHIVER.grid_sol_loc = (int*)(malloc(sizeof(int) * (ARCHIVER__MAX_SIZE + ARCHIVER__MAX_NEW_SOLS)));

    for (int i = 0; i < ARCHIVER__OBJECTIVES; i++) {
        ARCHIVER.gl_offset[i] = 0.0;
        ARCHIVER.gl_range[i] = 0.0;
        ARCHIVER.gl_largest[i] = 0.0;
    }

    for (int i = 0; i < ARCHIVER.max_locations; i++) {
        ARCHIVER.grid_pop[i] = 0;
    }

    #ifdef DEBUG_1
        fprintf(stderr, "[DEBUG] Archiver init\n");
        fprintf(stderr, "> population_size   : %d\n", ARCHIVER__MAX_SIZE);
    #endif
}

void archivers_aga_free(int new_solutions_size) {
    for (int p = 0; p < ARCHIVER__MAX_SIZE; p++) {
        if (ARCHIVER.population[p].initialized != SOLUTION__NOT_INITIALIZED) {
            free_solution(&ARCHIVER.population[p]);
        }
    }

    for (int i = 0; i < ARCHIVER__MAX_NEW_SOLS; i++) {
        free_solution(&ARCHIVER.new_solutions[i]);
    }

    free(ARCHIVER.grid_pop);
    free(ARCHIVER.grid_sol_loc);
}

int delete_dominated_solutions(int new_solutions_size) {
    // Elimino las soluciones dominadas entre si de la población de nuevas soluciones.
    for (int n_pos = 0; n_pos < new_solutions_size; n_pos++)
    {
        if (ARCHIVER.new_solutions[n_pos].initialized == SOLUTION__IN_USE)
        {
            for (int n2_pos = n_pos+1; (n2_pos < new_solutions_size) &&
                (ARCHIVER.new_solutions[n_pos].initialized == SOLUTION__IN_USE); n2_pos++)
            {
                if (ARCHIVER.new_solutions[n2_pos].initialized == SOLUTION__IN_USE)
                {
                    // Calculo no dominancia del elemento nuevo con el actual.
                    if ((ARCHIVER.new_solutions[n2_pos].makespan <= ARCHIVER.new_solutions[n_pos].makespan) &&
                        (ARCHIVER.new_solutions[n2_pos].energy_consumption <= ARCHIVER.new_solutions[n_pos].energy_consumption))
                    {
                        ARCHIVER.new_solutions[n_pos].initialized = SOLUTION__EMPTY;
                    }
                    else if ((ARCHIVER.new_solutions[n_pos].makespan <= ARCHIVER.new_solutions[n2_pos].makespan) &&
                        (ARCHIVER.new_solutions[n_pos].energy_consumption <= ARCHIVER.new_solutions[n2_pos].energy_consumption))
                    {
                        ARCHIVER.new_solutions[n2_pos].initialized = SOLUTION__EMPTY;
                    }
                }
            }
        }
    }

    int cant_new_solutions = 0;

    // Comparo la dominancia de las nuevas soluciones con las viejas soluciones.
    for (int n_pos = 0; n_pos < new_solutions_size; n_pos++)
    {
        if (ARCHIVER.new_solutions[n_pos].initialized == SOLUTION__IN_USE)
        {
            for (int s_pos = 0; (s_pos < ARCHIVER__MAX_SIZE) &&
                (ARCHIVER.new_solutions[n_pos].initialized == SOLUTION__IN_USE); s_pos++)
            {
                if (ARCHIVER.population[s_pos].initialized == SOLUTION__IN_USE)
                {
                    // Calculo no dominancia del elemento nuevo con el actual.
                    if ((ARCHIVER.population[s_pos].makespan <= ARCHIVER.new_solutions[n_pos].makespan) &&
                        (ARCHIVER.population[s_pos].energy_consumption <= ARCHIVER.new_solutions[n_pos].energy_consumption))
                    {
                        // La nueva solucion es dominada por una ya existente.
                        ARCHIVER.new_solutions[n_pos].initialized = SOLUTION__EMPTY;
                    }
                    else if ((ARCHIVER.new_solutions[n_pos].makespan <= ARCHIVER.population[s_pos].makespan) &&
                        (ARCHIVER.new_solutions[n_pos].energy_consumption <= ARCHIVER.population[s_pos].energy_consumption))
                    {
                        // La nueva solucion domina a una ya existente.
                        ARCHIVER.population[s_pos].initialized = SOLUTION__EMPTY;
                        ARCHIVER.population_count--;
                        ARCHIVER.tag_count[ARCHIVER.population_tag[s_pos]]--;
                    }
                }
            }
        }

        if (ARCHIVER.new_solutions[n_pos].initialized == SOLUTION__IN_USE)
        {
            cant_new_solutions++;
        }
    }

    return cant_new_solutions;
}

void archive_add_all_new(int new_solutions_size) {
    for (int n_pos = 0; n_pos < new_solutions_size; n_pos++)
    {
        if (ARCHIVER.new_solutions[n_pos].initialized == SOLUTION__IN_USE) {
            for (int s_pos = 0; (s_pos < ARCHIVER__MAX_SIZE) &&
                (ARCHIVER.new_solutions[n_pos].initialized == SOLUTION__IN_USE); s_pos++)
            {
                if (ARCHIVER.population[s_pos].initialized != SOLUTION__IN_USE)
                {
                    if (ARCHIVER.population[s_pos].initialized == SOLUTION__NOT_INITIALIZED) {
                        create_empty_solution(&ARCHIVER.population[s_pos]);
                    }

                    clone_solution(&ARCHIVER.population[s_pos], &ARCHIVER.new_solutions[n_pos]);

                    ARCHIVER.population_tag[s_pos] = ARCHIVER.new_solutions_tag[n_pos];
                    ARCHIVER.tag_count[ARCHIVER.population_tag[s_pos]]++;
                    ARCHIVER.population_count++;
                    ARCHIVER.new_solutions[n_pos].initialized = SOLUTION__EMPTY;
                }
            }
        }
    }
}

int archivers_aga(int new_solutions_size)
{
    assert(new_solutions_size <= ARCHIVER__MAX_NEW_SOLS);

    /*#ifdef DEBUG_3
        fprintf(stderr, "[DEBUG] archiver aga...\n");
    #endif*/

    int nd_new_solutions = delete_dominated_solutions(new_solutions_size);

    if (nd_new_solutions + ARCHIVER.population_count <= ARCHIVER__MAX_SIZE) {
        archive_add_all_new(new_solutions_size);
    } else {
        // Calculate grid location of mutant solution and renormalize archive if necessary.
        update_grid(new_solutions_size);

        // Update the archive by removing all dominated individuals.
        nd_new_solutions = archive_add_solution(nd_new_solutions);
    }

    return nd_new_solutions;
}

void archivers_aga_dump() {
    fprintf(stdout, "%d\n", ARCHIVER.population_count);
    for (int i = 0; i < ARCHIVER__MAX_SIZE; i++) {
        if (ARCHIVER.population[i].initialized == SOLUTION__IN_USE) {
            for (int task_id = 0; task_id < INPUT.tasks_count; task_id++) {
                fprintf(stdout, "%d\n", ARCHIVER.population[i].task_assignment[task_id]);
            }
        }
    }
}

void archivers_aga_show() {
    int pos = 0;

    for (int i = 0; i < ARCHIVER__MAX_SIZE; i++) {
        if (ARCHIVER.population[i].initialized == SOLUTION__IN_USE) {
            fprintf(stderr, "(%d) %f %f %d\n", pos++,
                ARCHIVER.population[i].makespan,
                ARCHIVER.population[i].energy_consumption,
                ARCHIVER.population_tag[i]);
        }
    }
    fprintf(stderr, "[DEBUG] Total solutions: %d\n", ARCHIVER.population_count);
}
