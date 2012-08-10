#ifndef LOAD_PARAMS_H_
#define LOAD_PARAMS_H_

#define ALGORITHM_MCT       0
#define ALGORITHM_MINMIN    1
#define ALGORITHM_PMINMIND  2

struct params {
    char *scenario_path;
    char *workload_path;
    
    int machines_count;
    int tasks_count;
       
    int seed;
    int thread_count;
    int algorithm;
    
    int max_time_secs;
    int max_iterations;
    int population_size;
};

int load_params(int argc, char **argv, struct params *input);

#endif /* LOAD_PARAMS_H_ */
