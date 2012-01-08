#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

#include "MersenneTwister.h"

int main(int argc, char** argv) {
    mersenne_twister_init_data init;

    char *data_path = "/home/siturria/cuda/palsGPU-MT/src/random/mersenne_twister/data/";
    mersenne_twister_init(data_path, 15, init_data);
    mersenne_twister_generate(init_data, 777);
    
    float results[15];
    
    mersenne_twister_read_results(init_data, results);
    mersenne_twister_free(init_data);
}
