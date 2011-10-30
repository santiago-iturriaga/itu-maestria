#NVCC=/home/santiago/cuda/bin/nvcc
NVCC=/usr/local/cuda/bin/nvcc 
NVCCLocal=/home/santiago/cuda/bin/nvcc 

all: pals

pals: src/main.cu src/pals_gpu.cu src/pals_serial.cu src/load_params.cu \
		src/load_instance.cu src/etc_matrix.cu src/solution.cu src/mct.cu
	$(NVCC) -g --ptxas-options=-v  src/pals_gpu.cu src/pals_serial.cu src/load_instance.cu \
		src/load_params.cu src/main.cu src/etc_matrix.cu src/solution.cu src/mct.cu -o bin/pals

local: src/main.cu src/pals_serial.cu src/pals_gpu.cu src/load_params.cu \
		src/load_instance.cu src/etc_matrix.cu src/solution.cu src/mct.cu
	$(NVCCLocal) -g --ptxas-options=-v  src/pals_gpu.cu src/pals_serial.cu src/load_instance.cu \
		src/load_params.cu src/main.cu src/etc_matrix.cu src/solution.cu src/mct.cu -o bin/pals
		
mem: src/pals_gpu.cu
	$(NVCCLocal) -ptx src/pals_gpu.cu -o mem_gpu.txt
		
clean: bin/pals
	rm bin/pals

