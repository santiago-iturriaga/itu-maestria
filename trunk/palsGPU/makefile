#CC=gcc

NVCC=/home/clusterusers/siturriaga/cuda/bin/nvcc -arch=compute_20
#NVCC=/usr/local/cuda/bin/nvcc 
NVCCLocal=/home/santiago/cuda/bin/nvcc -L/home/santiago/cuda/fake-libs/ -arch=compute_20

MTGP32_INC = src/random/mtgp-1.1/mtgp32-fast.h \
	src/random/mtgp-1.1/mtgp32-fast.c \
	src/random/mtgp-1.1/mtgp-util.h \
	src/random/mtgp-1.1/mtgp-util.cu \
	src/random/mtgp-1.1/mtgp32-cuda.cu \
	src/random/mtgp-1.1/mtgp32-cuda.h \
	src/random/mtgp-1.1/mtgp32dc-param-11213.c \
	src/cuda-util.h \
	src/cuda-util.cu
MTGP32_SRC = src/random/mtgp-1.1/mtgp32-fast.c \
	src/random/mtgp-1.1/mtgp-util.cu \
	src/random/mtgp-1.1/mtgp32-cuda.cu \
	src/random/mtgp-1.1/mtgp32dc-param-11213.c \
	src/cuda-util.cu
CUDALINK = -lcuda

all: pals-opt verificador minmin

verificador: verificador.c
	$(CC) -g verificador.c -o bin/verificador
	
minmin: minmin.c
	$(CC) -O3 minmin.c -o bin/minmin

pals-gdb: src/main.cu \
		src/load_params.cu src/load_instance.cu src/etc_matrix.cu \
		src/solution.cu src/utils.cu src/gpu_utils.cu \
		src/basic/mct.cu \
		src/basic/minmin.cu \
		src/random/cpu_rand.cu \
		src/random/RNG_rand48.cu \
		src/random/RNG_rand48.cu \
		src/pals/pals_gpu.cu \
		src/pals/pals_gpu_rtask.cu \
		src/pals/pals_serial.cu \
		${MTGP32_INC} 
	$(NVCC) -g --ptxas-options=-v -maxrregcount=16 src/main.cu \
		src/load_instance.cu src/load_params.cu src/etc_matrix.cu \
		src/solution.cu src/utils.cu src/gpu_utils.cu \
		src/basic/mct.cu \
		src/basic/minmin.cu \
		src/random/cpu_rand.cu \
		src/random/RNG_rand48.cu \
		src/pals/pals_gpu.cu \
		src/pals/pals_gpu_rtask.cu \
		src/pals/pals_serial.cu \
		${MTGP32_SRC} ${CUDALINK} \
			-o bin/pals

pals-opt: src/main.cu \
		src/load_params.cu src/load_instance.cu src/etc_matrix.cu \
		src/solution.cu src/utils.cu src/gpu_utils.cu \
		src/basic/mct.cu \
		src/basic/minmin.cu \
		src/random/cpu_rand.cu \
		src/random/RNG_rand48.cu \
		src/pals/pals_gpu.cu \
		src/pals/pals_gpu_rtask.cu \
		src/pals/pals_serial.cu \
		${MTGP32_INC} 
	$(NVCC) -O3 -maxrregcount=16 src/main.cu \
		src/load_instance.cu src/load_params.cu src/etc_matrix.cu \
		src/solution.cu src/utils.cu src/gpu_utils.cu \
		src/basic/mct.cu \
		src/basic/minmin.cu \
		src/random/cpu_rand.cu \
		src/random/RNG_rand48.cu \
		src/pals/pals_gpu.cu \
		src/pals/pals_gpu_rtask.cu \
		src/pals/pals_serial.cu \
		${MTGP32_SRC} ${CUDALINK} \
			-o bin/pals

pals-local: src/main.cu \
		src/load_params.cu src/load_instance.cu src/etc_matrix.cu \
		src/solution.cu src/utils.cu src/gpu_utils.cu \
		src/basic/mct.cu \
		src/basic/minmin.cu \
		src/random/cpu_rand.cu \
		src/random/RNG_rand48.cu \
		src/pals/pals_gpu.cu \
		src/pals/pals_gpu_rtask.cu \
		src/pals/pals_serial.cu \
		${MTGP32_INC} 
	$(NVCCLocal) -g --ptxas-options=-v -maxrregcount=16 src/main.cu \
		src/load_instance.cu src/load_params.cu src/etc_matrix.cu \
		src/solution.cu src/utils.cu src/gpu_utils.cu \
		src/basic/mct.cu \
		src/basic/minmin.cu \
		src/random/cpu_rand.cu \
		src/random/RNG_rand48.cu \
		src/pals/pals_gpu.cu \
		src/pals/pals_gpu_rtask.cu \
		src/pals/pals_serial.cu \
		${MTGP32_SRC} ${CUDALINK} \
			-o bin/pals

clean:
	rm bin/*
