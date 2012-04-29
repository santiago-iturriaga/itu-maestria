#CC=gcc

NVCC=/home/clusterusers/siturriaga/cuda/bin/nvcc
#NVCC=/usr/local/cuda/bin/nvcc 
NVCCLocal=/home/santiago/cuda/bin/nvcc 

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
		src/pals/pals_gpu.cu \
		src/pals/pals_gpu_rtask.cu \
		src/pals/pals_gpu_prtask.cu \
		src/pals/pals_serial.cu 
	$(NVCCLocal) -g --ptxas-options=-v -maxrregcount=16 src/main.cu \
		src/load_instance.cu src/load_params.cu src/etc_matrix.cu \
		src/solution.cu src/utils.cu src/gpu_utils.cu \
		src/basic/mct.cu \
		src/basic/minmin.cu \
		src/random/cpu_rand.cu \
		src/random/RNG_rand48.cu \
		src/pals/pals_gpu.cu \
		src/pals/pals_gpu_rtask.cu \
		src/pals/pals_gpu_prtask.cu \
		src/pals/pals_serial.cu \
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
		src/pals/pals_gpu_prtask.cu \
		src/pals/pals_serial.cu 
	$(NVCC) -maxrregcount=16 src/main.cu \
		src/load_instance.cu src/load_params.cu src/etc_matrix.cu \
		src/solution.cu src/utils.cu src/gpu_utils.cu \
		src/basic/mct.cu \
		src/basic/minmin.cu \
		src/random/cpu_rand.cu \
		src/random/RNG_rand48.cu \
		src/pals/pals_gpu.cu \
		src/pals/pals_gpu_rtask.cu \
		src/pals/pals_gpu_prtask.cu \
		src/pals/pals_serial.cu \
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
		src/pals/pals_gpu_prtask.cu \
		src/pals/pals_serial.cu 
	$(NVCCLocal) -g --ptxas-options=-v -maxrregcount=16 src/main.cu \
		src/load_instance.cu src/load_params.cu src/etc_matrix.cu \
		src/solution.cu src/utils.cu src/gpu_utils.cu \
		src/basic/mct.cu \
		src/basic/minmin.cu \
		src/random/cpu_rand.cu \
		src/random/RNG_rand48.cu \
		src/pals/pals_gpu.cu \
		src/pals/pals_gpu_rtask.cu \
		src/pals/pals_gpu_prtask.cu \
		src/pals/pals_serial.cu \
			-o bin/pals

clean:
	rm bin/*

