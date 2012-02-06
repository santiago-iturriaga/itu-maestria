CC=gcc -Wall -std=c99
CX=g++ -Wall
LIBS=-lpthread -lrt
OUTPUT_BIN=pals_cpu_speed

all: pals-opt verificador fp

fp: fp_2obj.cpp
	$(CXX) fp_2obj.cpp -o bin/fp_2obj

verificador: verificador.c
	$(CC) verificador.c -o bin/verificador
	
pals-gdb: src/main.cpp \
		src/load_params.cpp \
		src/load_instance.cpp \
		src/etc_matrix.cpp \
		src/energy_matrix.cpp \
		src/solution.cpp \
		src/utils.cpp \
		src/basic/mct.cpp \
		src/basic/minmin.cpp \
		src/random/cpu_rand.cpp \
		src/random/cpu_mt.cpp \
		src/pals/pals_cpu_2pop.cpp \
		src/pals/pals_cpu_1pop.cpp \
		src/pals/pals_serial.cpp 
	$(CX) -g src/main.cpp \
		src/load_instance.cpp \
		src/load_params.cpp \
		src/etc_matrix.cpp \
		src/energy_matrix.cpp \
		src/solution.cpp \
		src/utils.cpp \
		src/basic/mct.cpp \
		src/basic/minmin.cpp \
		src/random/cpu_rand.cpp \
		src/random/cpu_mt.cpp \
		src/pals/pals_cpu_2pop.cpp \
		src/pals/pals_cpu_1pop.cpp \
		src/pals/pals_serial.cpp \
			-o bin/$(OUTPUT_BIN) $(LIBS)

pals-opt: src/main.cpp \
		src/load_params.cpp \
		src/load_instance.cpp \
		src/etc_matrix.cpp \
		src/energy_matrix.cpp \
		src/solution.cpp \
		src/utils.cpp \
		src/basic/mct.cpp \
		src/basic/minmin.cpp \
		src/random/cpu_rand.cpp \
		src/random/cpu_mt.cpp \
		src/pals/pals_cpu_2pop.cpp \
		src/pals/pals_cpu_1pop.cpp \
		src/pals/pals_serial.cpp 
	$(CX) -O3 src/main.cpp \
		src/load_instance.cpp \
		src/load_params.cpp \
		src/etc_matrix.cpp \
		src/energy_matrix.cpp \
		src/solution.cpp \
		src/utils.cpp \
		src/basic/mct.cpp \
		src/basic/minmin.cpp \
		src/random/cpu_rand.cpp \
		src/random/cpu_mt.cpp \
		src/pals/pals_cpu_2pop.cpp \
		src/pals/pals_cpu_1pop.cpp \
		src/pals/pals_serial.cpp \
			-o bin/$(OUTPUT_BIN) $(LIBS)

clean:
	rm bin/$(OUTPUT_BIN)

