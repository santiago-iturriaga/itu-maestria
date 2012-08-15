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

CPU_BIN = gmochc_cpu
CPU_LINK = -lrt -lpthread -lm
CPU_OPTS = -Wall -DCPU -fno-exceptions

BIN = gmochc
LINK = -lcuda -lrt -lpthread -lm
OPTS = --ptxas-options=-v

all: debug verificador minmin

verificador: verificador.c
	$(CC) $(CPU_OPTS) -g verificador.c -o bin/verificador

minmin: minmin.c
	$(CC) $(CPU_OPTS) -O3 minmin.c -o bin/minmin

cpu: src/main.cu \
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
		src/random/cpu_rand.h \
		src/random/cpu_rand.cu \
		src/random/cpu_drand48.h \
		src/random/cpu_drand48.cu \
		src/random/cpu_mt.h \
		src/random/cpu_mt.cu \
		src/cmochc/cmochc_cell.h \
		src/cmochc/cmochc_cell.cu \
		src/cmochc/cmochc_island.h \
		src/cmochc/cmochc_island.cu \
		src/basic/mct.cu \
		src/basic/mct.h \
		src/basic/minmin.cu \
		src/basic/minmin.h \
		src/basic/pminmin.cu \
		src/basic/pminmin.h 
	$(CC) $(CPU_OPTS) -g -x c++ \
		src/main.cu \
		src/load_instance.cu \
		src/load_params.cu \
		src/scenario.cu \
		src/etc_matrix.cu \
		src/energy_matrix.cu \
		src/solution.cu \
		src/utils.cu \
		src/random/cpu_rand.cu \
		src/random/cpu_drand48.cu \
		src/random/cpu_mt.cu \
		src/cmochc/cmochc_cell.cu \
		src/cmochc/cmochc_island.cu \
		src/basic/mct.cu \
		src/basic/minmin.cu \
		src/basic/pminmin.cu \
		$(CPU_LINK) -o bin/$(CPU_BIN)

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
		src/cmochc/cmochc_cell.h \
		src/cmochc/cmochc_cell.cu \
		src/cmochc/cmochc_island.h \
		src/cmochc/cmochc_island.cu \
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
	$(NVCC) $(OPTS) -g \
		src/main.cu \
		src/load_instance.cu \
		src/load_params.cu \
		src/scenario.cu \
		src/etc_matrix.cu \
		src/energy_matrix.cu \
		src/solution.cu \
		src/utils.cu \
		src/cuda_utils.cu \
		src/cmochc/cmochc_cell.cu \
		src/cmochc/cmochc_island.cu \
		src/basic/mct.cu \
		src/basic/minmin.cu \
		src/basic/pminmin.cu \
		src/random/cpu_rand.cu \
		src/random/RNG_rand48.cu \
		$(MTGP32_SRC) $(LINK) \
		-o bin/$(BIN)

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
		src/cmochc/cmochc_cell.h \
		src/cmochc/cmochc_cell.cu \
		src/cmochc/cmochc_island.h \
		src/cmochc/cmochc_island.cu \
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
		src/cmochc/cmochc_cell.cu \
		src/cmochc/cmochc_island.cu \
		src/basic/mct.cu \
		src/basic/minmin.cu \
		src/basic/pminmin.cu \
		src/random/cpu_rand.cu \
		src/random/RNG_rand48.cu \
		$(MTGP32_SRC) $(LINK) \
		-o bin/$(BIN)

clean:
	rm bin/*
