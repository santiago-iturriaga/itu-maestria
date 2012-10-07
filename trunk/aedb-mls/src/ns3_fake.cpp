#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

int main(int argc, char* argv[]) {
    assert(argc == 8);
    
    int number_devices = atoi(argv[1]);
    fprintf(stderr, "number_devices     : %d\n", number_devices);
    
    int simul_runs = atoi(argv[2]);
    fprintf(stderr, "simul_runs         : %d\n", simul_runs);
    
    double min_delay = atof(argv[3]);
    fprintf(stderr, "min_delay          : %f\n", min_delay);
    
    double max_delay = atof(argv[4]);
    fprintf(stderr, "max_delay          : %f\n", max_delay);
    
    double borders_threshold = atof(argv[5]);
    fprintf(stderr, "borders_threshold  : %f\n", borders_threshold);
    
    double margin_forwarding = atof(argv[6]);
    fprintf(stderr, "margin_forwarding  : %f\n", margin_forwarding);
    
    int neighbors_threshold = atoi(argv[7]);
    fprintf(stderr, "neighbors_threshold: %d\n", neighbors_threshold);
    
    srand(time(NULL));
    
    fprintf(stdout, "%f\n", (double)rand() / (double)RAND_MAX);
    fprintf(stdout, "%f\n", (double)rand() / (double)RAND_MAX);
    fprintf(stdout, "%f\n", (double)rand() / (double)RAND_MAX);
    fprintf(stdout, "%f\n", (double)rand() / (double)RAND_MAX);
    
    return EXIT_SUCCESS;
}
