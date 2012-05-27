#DEBUG = -DDEBUG -g -O0
#ARCH = -arch=compute_11
#ARCH = -arch=compute_13
#ARCH = -arch=compute_20
VERB = -Xptxas=-v --ptxas-options=-v
OMP = -O3 -Xcompiler "-g -pg -fopenmp -lgomp"

MTGP32_INC = src/mtgp-1.1/mtgp32-fast.h src/mtgp-1.1/mtgp32-fast.c src/mtgp-1.1/mtgp-util.h \
	src/mtgp-1.1/mtgp-util.cu src/mtgp-1.1/mtgp32-cuda.cu src/mtgp-1.1/mtgp32-cuda.h \
	src/mtgp-1.1/mtgp32dc-param-11213.c src/cuda-util.h src/cuda-util.cu src/util.h
MTGP32_SRC = src/mtgp-1.1/mtgp32-fast.c src/mtgp-1.1/mtgp-util.cu src/mtgp-1.1/mtgp32-cuda.cu \
	src/mtgp-1.1/mtgp32dc-param-11213.c src/cuda-util.cu
CUDALINK = -lcuda

NVCC_Marga = /home/santiago/cuda/bin/nvcc ${DEBUG} ${ARCH} -D__STDC_FORMAT_MACROS \
	-L/home/santiago/cuda/fake-libs/ -D__STDC_CONSTANT_MACROS ${VERB}
NVCC_CLu = /home/clusterusers/siturriaga/cuda/bin/nvcc ${DEBUG} ${ARCH} \
	-D__STDC_FORMAT_MACROS -D__STDC_CONSTANT_MACROS ${VERB}
NVCC = /usr/local/cuda/bin/nvcc ${DEBUG} ${ARCH} -D__STDC_FORMAT_MACROS -D__STDC_CONSTANT_MACROS ${VERB}

all: clu

marga: src/main.cu src/billionga.cu ${MTGP32_INC} 
	$(NVCC_Marga) -o bin/billionga src/main.cu src/billionga.cu \
	${MTGP32_SRC} ${CUDALINK} ${OMP}

clu: src/main.cu src/billionga.cu ${MTGP32_INC} 
	$(NVCC_CLu) -o bin/billionga src/main.cu src/billionga.cu \
	${MTGP32_SRC} ${CUDALINK} ${OMP}

cuy: src/main.cu src/billionga.cu ${MTGP32_INC} 
	$(NVCC) -o bin/billionga src/main.cu src/billionga.cu \
	${MTGP32_SRC} ${CUDALINK} ${OMP}

omp: src/main.cu src/billionga.cu ${MTGP32_INC} 
	$(NVCC_CLu) -o bin/omp_main omp_main.cu src/billionga.cu \
	${MTGP32_SRC} ${CUDALINK} ${OMP}

test: src/test.c
	$(CC) -o bin/test src/test.c -std=c99 -g
