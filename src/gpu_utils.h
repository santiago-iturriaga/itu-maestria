#include <cuda.h>

#ifndef GPU_UTILS__
#define GPU_UTILS__

// Obtiene la cantidad de tarjetas disponibles en el sistema.
int gpu_get_devices_count();

// Obtiene las propiedades de las tarjetas del sistema.
void gpu_get_devices(cudaDeviceProp devices[], int &size);

// Muestra los datos de las tarjetas del sistema.
void gpu_show_devices();

// Establece la tarjeta a utilizar.
void gpu_set_device(int device);

// Establece la "mejor" tarjeta para utilizar.
void gpu_set_best_device();

#endif
