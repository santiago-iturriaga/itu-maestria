/*
 * load_params.h
 *
 *  Created on: Jul 27, 2011
 *      Author: santiago
 */

#ifndef LOAD_PARAMS_H_
#define LOAD_PARAMS_H_

#define PALS_2POP 0
#define PALS_1POP 1
#define    MINMIN 2
#define       MCT 3

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
};

int load_params(int argc, char **argv, struct params *input);

#endif /* LOAD_PARAMS_H_ */
