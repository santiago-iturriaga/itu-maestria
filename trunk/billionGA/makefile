NVCC_Dora=/home/santiago/cuda/bin/nvcc
NVCC=nvcc

all: dora-test-mt 
#dora-debug

dora-debug: src/main.cu src/billionga.cu src/mt/MersenneTwister.cu
	$(NVCC_Dora) -g --ptxas-options=-v -maxrregcount=16 \
		src/main.cu src/billionga.cu src/mt/MersenneTwister.cu \
		-o bin/billionga

dora-test-mt: src/mt/prueba.cu src/mt/MersenneTwister.cu
	$(NVCC_Dora) -g --ptxas-options=-v -maxrregcount=16 \
		src/mt/prueba.cu src/mt/MersenneTwister.cu -o bin/test_mt
