CC=mpicc -Wall -std=c99
CX=mpicxx -Wall
LIBS=-lpthread -lrt 
OUTPUT_BIN=mls

all: mls

mls-gdb: src/main.cpp src/config.h \
		src/solution.h \
		src/solution.cpp \
		src/random/cpu_mt.h \
		src/random/cpu_mt.cpp \
		src/aga/aga.h \
		src/aga/aga.cpp \
		src/mls/mls.h \
		src/mls/mls.cpp 
	$(CX) -g src/main.cpp \
		src/solution.cpp \
		src/random/cpu_mt.cpp \
		src/aga/aga.cpp \
		src/mls/mls.cpp \
		$(LIBS) -o bin/$(OUTPUT_BIN) 

mls: src/main.cpp src/config.h \
		src/solution.h \
		src/solution.cpp \
		src/random/cpu_mt.h \
		src/random/cpu_mt.cpp \
		src/aga/aga.h \
		src/aga/aga.cpp \
		src/mls/mls.h \
		src/mls/mls.cpp  
	$(CX) -O3 src/main.cpp \
		src/solution.cpp \
		src/random/cpu_mt.cpp \
		src/aga/aga.cpp \
		src/mls/mls.cpp \
		$(LIBS) -o bin/$(OUTPUT_BIN)

clean:
	rm bin/$(OUTPUT_BIN)
