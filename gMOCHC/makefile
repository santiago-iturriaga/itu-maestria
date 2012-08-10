#NVCC = /home/clusterusers/siturriaga/cuda/bin/nvcc
#NVCC = /usr/local/cuda/bin/nvcc 
NVCC = /home/santiago/cuda/bin/nvcc -L/home/santiago/cuda/fake-libs/

MTGP32_INC = src/random/mtgp-1.1/mtgp32-fast.h \
	src/random/mtgp-1.1/mtgp32-fast.c \
	src/random/mtgp-1.1/mtgp-util.h \
	src/random/mtgp-1.1/mtgp-util.cu \
	src/random/mtgp-1.1/mtgp32-cuda.cu \
	src/random/mtgp-1.1/mtgp32-cuda.h \
	src/random/mtgp-1.1/mtgp32dc-param-11213.c
MTGP32_SRC = src/random/mtgp-1.1/mtgp32-fast.c \
	src/random/mtgp-1.1/mtgp-util.cu \
	src/random/mtgp-1.1/mtgp32-cuda.cu \
	src/random/mtgp-1.1/mtgp32dc-param-11213.c

LINK = -lcuda -lrt -lpthread
#OPTS = -pedantic -wall
OPTS = 

all: debug verificador minmin

verificador: verificador.c
	$(CC) $(OPTS) -g verificador.c -o bin/verificador

minmin: minmin.c
	$(CC) $(OPTS) -O3 minmin.c -o bin/minmin

debug: src/main.cu \
		src/load_params.cu \
		src/load_params.h \
		src/load_instance.cu \
		src/load_instance.h \
		src/scenario.cu \
		src/scenario.h \
		src/etc_matrix.cu \
		src/etc_matrix.h \
		src/energy_matrix.cu \
		src/energy_matrix.h \
		src/solution.cu \
		src/solution.h \
		src/utils.cu \
		src/utils.h \
		src/cuda_utils.cu \
		src/cuda_utils.h \
		src/basic/mct.cu \
		src/basic/mct.h \
		src/basic/minmin.cu \
		src/basic/minmin.h \
		src/basic/pminmin.cu \
		src/basic/pminmin.h \
		src/random/cpu_rand.cu \
		src/random/cpu_rand.h \
		src/random/RNG_rand48.cu \
		src/random/RNG_rand48.h \
		${MTGP32_INC} 
	$(NVCC) $(OPTS) -g --ptxas-options=-v \
		src/main.cu \
		src/load_instance.cu \
		src/load_params.cu \
		src/scenario.cu \
		src/etc_matrix.cu \
		src/energy_matrix.cu \
		src/solution.cu \
		src/utils.cu \
		src/cuda_utils.cu \
		src/basic/mct.cu \
		src/basic/minmin.cu \
		src/basic/pminmin.cu \
		src/random/cpu_rand.cu \
		src/random/RNG_rand48.cu \
		${MTGP32_SRC} ${LINK} \
		-o bin/gmochc

release: src/main.cu \
		src/load_params.cu \
		src/load_params.h \
		src/load_instance.cu \
		src/load_instance.h \
		src/scenario.cu \
		src/scenario.h \
		src/etc_matrix.cu \
		src/etc_matrix.h \
		src/energy_matrix.cu \
		src/energy_matrix.h \
		src/solution.cu \
		src/solution.h \
		src/utils.cu \
		src/utils.h \
		src/cuda_utils.cu \
		src/cuda_utils.h \
		src/basic/mct.cu \
		src/basic/mct.h \
		src/basic/minmin.cu \
		src/basic/minmin.h \
		src/basic/pminmin.cu \
		src/basic/pminmin.h \
		src/random/cpu_rand.cu \
		src/random/cpu_rand.h \
		src/random/RNG_rand48.cu \
		src/random/RNG_rand48.h \
		${MTGP32_INC} 
	$(NVCC) $(OPTS) -O3 \
		src/main.cu \
		src/load_instance.cu \
		src/load_params.cu \
		src/scenario.cu \
		src/etc_matrix.cu \
		src/energy_matrix.cu \
		src/solution.cu \
		src/utils.cu \
		src/cuda_utils.cu \
		src/basic/mct.cu \
		src/basic/minmin.cu \
		src/basic/pminmin.cu \
		src/random/cpu_rand.cu \
		src/random/RNG_rand48.cu \
		${MTGP32_SRC} ${LINK} \
		-o bin/gmochc 

clean:
	rm bin/*
