CC=~/bin/mpich2-1.4.1p1/bin/mpicc -Wall -std=c99
CX=~/bin/mpich2-1.4.1p1/bin/mpic++ -Wall
LIBS=-lpthread -lrt -I/home/clusterusers/siturriaga/ns3Files -lns3 -L/home/clusterusers/siturriaga/ns3Files

CC_LOCAL=mpicc -Wall -std=c99
CX_LOCAL=mpic++ -Wall
LIBS_LOCAL=-lpthread -lrt -DLOCAL

OUTPUT_BIN=mls

all: mls

mls-local: src/main.cpp src/config.h \
		src/solution.h \
		src/solution.cpp \
		src/random/cpu_mt.h \
		src/random/cpu_mt.cpp \
		src/aga/aga.h \
		src/aga/aga.cpp \
		src/mls/mls.h \
		src/mls/mls.cpp  
	$(CX_LOCAL) -g src/main.cpp \
		src/solution.cpp \
		src/random/cpu_mt.cpp \
		src/aga/aga.cpp \
		src/mls/mls.cpp \
		$(LIBS_LOCAL) -o bin/$(OUTPUT_BIN) 

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
	$(CX) -O3 -DNDEBUG src/main.cpp \
		src/solution.cpp \
		src/random/cpu_mt.cpp \
		src/aga/aga.cpp \
		src/mls/mls.cpp \
		$(LIBS) -o bin/$(OUTPUT_BIN)

clean:
	rm bin/$(OUTPUT_BIN)
