#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <mpi.h>

#include "config.h"
#include "solution.h"

MPI_Datatype mpi_solution_type;

void init_mpi_solution() {
    struct solution aux;

    int lengtharray[10];         /* Array of lengths */
    MPI_Aint disparray[10];      /* Array of displacements */
    MPI_Datatype typearray[10];  /* Array of MPI datatypes */

    MPI_Aint startaddress, address;   /* Variables used to calculate displacements */

    /* Set array of lengths */
    lengtharray[0] = 1;
    lengtharray[1] = 1;
    lengtharray[2] = 1;
    lengtharray[3] = 1;
    lengtharray[4] = 1;
    lengtharray[5] = 1;
    lengtharray[6] = 1;
    lengtharray[7] = 1;
    lengtharray[8] = 1;
    lengtharray[9] = 1;

    /* And data types */
    typearray[0] = MPI_INT;
    typearray[1] = MPI_DOUBLE;
    typearray[2] = MPI_DOUBLE;
    typearray[3] = MPI_DOUBLE;
    typearray[4] = MPI_DOUBLE;
    typearray[5] = MPI_INT;
    typearray[6] = MPI_DOUBLE;
    typearray[7] = MPI_DOUBLE;
    typearray[8] = MPI_DOUBLE;
    typearray[9] = MPI_DOUBLE;
    
    /* First element is at displacement 0 */
    disparray[0] = 0;

    /* Calculate displacement of others */
    MPI_Address(&aux.status, &startaddress);
    MPI_Address(&aux.borders_threshold, &address);
    disparray[1] = address-startaddress;     

    MPI_Address(&aux.margin_forwarding, &address);
    disparray[2] = address-startaddress;     

    MPI_Address(&aux.min_delay, &address);
    disparray[3] = address-startaddress;     
    
    MPI_Address(&aux.max_delay, &address);
    disparray[4] = address-startaddress;   
    
    MPI_Address(&aux.neighbors_threshold, &address);
    disparray[5] = address-startaddress;     
    
    MPI_Address(&aux.energy, &address);
    disparray[6] = address-startaddress;     
    
    MPI_Address(&aux.coverage, &address);
    disparray[7] = address-startaddress;     
    
    MPI_Address(&aux.nforwardings, &address);
    disparray[8] = address-startaddress;     

    MPI_Address(&aux.time, &address);
    disparray[9] = address-startaddress;     

    /* Build the data structure */
    MPI_Type_struct(10, lengtharray, disparray, typearray, &mpi_solution_type);
    MPI_Type_commit(&mpi_solution_type);
}

void clone_solution(struct solution *dst, struct solution *src) {
    dst->status = src->status;

    dst->borders_threshold = src->borders_threshold;
    dst->margin_forwarding = src->margin_forwarding;
    dst->min_delay = src->min_delay;
    dst->max_delay = src->max_delay;
    dst->neighbors_threshold = src->neighbors_threshold;

    dst->energy = src->energy;
    dst->coverage = src->coverage;
    dst->nforwardings = src->nforwardings;
    dst->time = src->time;
}

void show_solution(struct solution *sol) {
    fprintf(stderr, "-> borders_threshold   : %.2f\n", sol->borders_threshold);
    fprintf(stderr, "   margin_forwarding   : %.2f\n", sol->margin_forwarding);
    fprintf(stderr, "   min_delay           : %.2f\n", sol->min_delay);
    fprintf(stderr, "   max_delay           : %.2f\n", sol->max_delay);
    fprintf(stderr, "   neighbors_threshold : %d\n", sol->neighbors_threshold);
    fprintf(stderr, "   energy              : %.2f\n", sol->energy);
    fprintf(stderr, "   coverage            : %.2f\n", sol->coverage);
    fprintf(stderr, "   nforwardings        : %.2f\n", sol->nforwardings);
    fprintf(stderr, "   time                : %.2f\n", sol->time);
}
