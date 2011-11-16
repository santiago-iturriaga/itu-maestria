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

struct params {
	char *instance_path;
	int machines_count;
	int tasks_count;
	
	int pals_flavour;
};

int load_params(int argc, char **argv, struct params *input);

#endif /* LOAD_PARAMS_H_ */
