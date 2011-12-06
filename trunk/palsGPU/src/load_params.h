/*
 * load_params.h
 *
 *  Created on: Jul 27, 2011
 *      Author: santiago
 */

#ifndef LOAD_PARAMS_H_
#define LOAD_PARAMS_H_

#define PALS_Serial 0
#define PALS_GPU 1
#define PALS_GPU_randTask 2
#define PALS_GPU_randMachine 3
#define MinMin 4
#define MCT 5
#define PALS_GPU_randParallelTask 6

struct params {
	char *instance_path;
	int machines_count;
	int tasks_count;
	int seed;
	int gpu_device;
	int algorithm;
};

int load_params(int argc, char **argv, struct params *input);

#endif /* LOAD_PARAMS_H_ */
