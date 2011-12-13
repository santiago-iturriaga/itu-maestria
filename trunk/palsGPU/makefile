#NVCC=/home/santiago/cuda/bin/nvcc
NVCC=/usr/local/cuda/bin/nvcc 
NVCCLocal=/home/santiago/cuda/bin/nvcc 

all: pals

pals: src/main.cu \
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
	$(NVCC) -g --ptxas-options=-v -maxrregcount=16 src/main.cu \
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

opt: src/main.cu \
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
	$(NVCC) -O3 -maxrregcount=16 src/main.cu \
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

local: src/main.cu \
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

clean: bin/pals
	rm bin/pals

