CC=gcc -Wall -std=c99
CX=g++ -Wall
LIBS=-lpthread -lrt 
OUTPUT_BIN=mls

all: pals-opt

pals-gdb: src/main.cpp \
		src/solution.cpp \
		src/random/cpu_mt.cpp \
		src/search/archivers/aga.cpp \
		src/search/mls.cpp 
	$(CX) -g src/main.cpp \
		src/solution.cpp \
		src/random/cpu_mt.cpp \
		src/search/archivers/aga.cpp \
		src/search/mls.cpp \
		$(LIBS) -o bin/$(OUTPUT_BIN) 

pals-opt: src/main.cpp \
		src/solution.cpp \
		src/random/cpu_mt.cpp \
		src/search/archivers/aga.cpp \
		src/search/mls.cpp 
	$(CX) -O3 src/main.cpp \
		src/solution.cpp \
		src/random/cpu_mt.cpp \
		src/search/archivers/aga.cpp \
		src/search/mls.cpp \
		$(LIBS) -o bin/$(OUTPUT_BIN)

clean:
	rm bin/$(OUTPUT_BIN)
