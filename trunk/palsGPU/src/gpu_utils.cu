#include <stdlib.h>
#include <stdio.h>

#include "gpu_utils.h"

int gpu_get_devices_count() {
	int deviceCount;

	if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Obteniendo la cantidad de dispositivos disponibles\n");
		exit(EXIT_FAILURE);
	}
	
	return deviceCount;
}

void gpu_get_devices(cudaDeviceProp devices[], int &size) {
	size = gpu_get_devices_count();

	for (int device = 0; device < size; ++device) {
		if (cudaGetDeviceProperties(&(devices[device]), device) != cudaSuccess) {
			fprintf(stderr, "[ERROR] Obteniendo las propiedades de dispositivo %d.\n", device);
			exit(EXIT_FAILURE);			
		}
	}
}

void gpu_show_devices() {
	/*
	struct cudaDeviceProp {
		char name[256];
		size_t totalGlobalMem;
		size_t sharedMemPerBlock;
		int regsPerBlock;
		int warpSize;
		size_t memPitch;
		int maxThreadsPerBlock;
		int maxThreadsDim[3];
		int maxGridSize[3];
		size_t totalConstMem;
		int major;
		int minor;
		int clockRate;
		size_t textureAlignment;
		int deviceOverlap;
		int multiProcessorCount;
	}
	*/
	
	int size = gpu_get_devices_count();
	cudaDeviceProp devices[size];
	
	gpu_get_devices(devices, size);
	
	for (int i = 0; i < size; i++) {
		fprintf(stdout, "[[-- DEVICE %d --]]\n", i);
		fprintf(stdout, "Name: %s\n", devices[i].name);
		fprintf(stdout, "Clock: %d\n", devices[i].clockRate);
	}
}

void gpu_set_device(int device) {
	if (cudaSetDevice(device) != cudaSuccess) {
		fprintf(stderr, "[ERROR] Estableciendo el dispositivo %d\n", device);
		exit(EXIT_FAILURE);
	}
}

void gpu_set_best_device() {

}
