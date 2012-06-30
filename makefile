CC=gcc -Wall -std=c99
CX=g++ -Wall
LIBS=-lpthread -lrt 
#-mtune=opteron -mfpmath=sse -m64
OUTPUT_BIN=pals_cpu

all: pals-opt verificador fp

fp: fp_2obj.cpp
	$(CXX) fp_2obj.cpp -o bin/fp_2obj

verificador: verificador.c
	$(CC) verificador.c -o bin/verificador
	

pals-prof: src/main.cpp \
		src/load_params.cpp \
		src/load_instance.cpp \
		src/etc_matrix.cpp \
		src/energy_matrix.cpp \
		src/solution.cpp \
		src/utils.cpp \
		src/basic/mct.cpp \
		src/basic/minmin.cpp \
		src/basic/pminmin.cpp \
		src/random/cpu_rand.cpp \
        src/random/cpu_drand48.cpp \
		src/random/cpu_mt.cpp \
        src/pals/archivers/aga.cpp \
		src/pals/pals_cpu_1pop.cpp 
	$(CX) -pg src/main.cpp \
		src/load_instance.cpp \
		src/load_params.cpp \
		src/etc_matrix.cpp \
		src/energy_matrix.cpp \
		src/solution.cpp \
		src/utils.cpp \
		src/basic/mct.cpp \
		src/basic/minmin.cpp \
		src/basic/pminmin.cpp \
		src/random/cpu_rand.cpp \
		src/random/cpu_drand48.cpp \
		src/random/cpu_mt.cpp \
		src/pals/archivers/aga.cpp \
		src/pals/pals_cpu_1pop.cpp \
		$(LIBS) -o bin/pals_cpu_prof

pals-gdb: src/main.cpp \
		src/load_params.cpp \
		src/load_instance.cpp \
		src/etc_matrix.cpp \
		src/energy_matrix.cpp \
		src/solution.cpp \
		src/utils.cpp \
		src/basic/mct.cpp \
		src/basic/minmin.cpp \
		src/basic/pminmin.cpp \
		src/random/cpu_rand.cpp \
        src/random/cpu_drand48.cpp \
		src/random/cpu_mt.cpp \
        src/pals/archivers/aga.cpp \
		src/pals/pals_cpu_1pop.cpp 
	$(CX) -g src/main.cpp \
		src/load_instance.cpp \
		src/load_params.cpp \
		src/etc_matrix.cpp \
		src/energy_matrix.cpp \
		src/solution.cpp \
		src/utils.cpp \
		src/basic/mct.cpp \
		src/basic/minmin.cpp \
		src/basic/pminmin.cpp \
		src/random/cpu_rand.cpp \
		src/random/cpu_drand48.cpp \
		src/random/cpu_mt.cpp \
		src/pals/archivers/aga.cpp \
		src/pals/pals_cpu_1pop.cpp \
		$(LIBS) -o bin/$(OUTPUT_BIN) 

pals-opt: src/main.cpp \
		src/load_params.cpp \
		src/load_instance.cpp \
		src/etc_matrix.cpp \
		src/energy_matrix.cpp \
		src/solution.cpp \
		src/utils.cpp \
		src/basic/mct.cpp \
		src/basic/minmin.cpp \
		src/basic/pminmin.cpp \
		src/random/cpu_rand.cpp \
		src/random/cpu_drand48.cpp \
		src/random/cpu_mt.cpp \
		src/pals/archivers/aga.cpp \
		src/pals/pals_cpu_1pop.cpp 
	$(CX) -O3 src/main.cpp \
		src/load_params.cpp \
		src/load_instance.cpp \
		src/etc_matrix.cpp \
		src/energy_matrix.cpp \
		src/solution.cpp \
		src/utils.cpp \
		src/basic/mct.cpp \
		src/basic/minmin.cpp \
		src/basic/pminmin.cpp \
		src/random/cpu_rand.cpp \
		src/random/cpu_drand48.cpp \
		src/random/cpu_mt.cpp \
		src/pals/archivers/aga.cpp \
		src/pals/pals_cpu_1pop.cpp \
		$(LIBS) -o bin/$(OUTPUT_BIN)

clean:
	rm bin/$(OUTPUT_BIN)
