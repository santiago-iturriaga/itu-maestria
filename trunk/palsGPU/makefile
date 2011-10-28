#NVCC=/home/santiago/cuda/bin/nvcc
NVCC=/usr/local/cuda/bin/nvcc 

all: pals_gpu

pals_gpu: src/main.cu src/pals.cu src/load_params.cu src/load_instance.cu \
		src/etc_matrix.cu src/solution.cu src/mct.cu
	$(NVCC) --ptxas-options=-v  src/pals.cu src/load_instance.cu src/load_params.cu src/main.cu \
		src/etc_matrix.cu src/solution.cu src/mct.cu -o bin/pals_gpu

clean: bin/pals_gpu
	rm bin/pals_gpu

