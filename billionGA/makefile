#DEBUG = -DDEBUG -g -O0

#ARCH = -arch=compute_11
#ARCH = -arch=compute_13
#VERB = -Xptxas=-v
#--ptxas-options=-v

MTGP32_INC = src/mtgp-1.1/mtgp32-fast.h src/mtgp-1.1/mtgp32-fast.c src/mtgp-1.1/mtgp-util.cuh src/mtgp-1.1/mtgp-util.cu src/mtgp-1.1/mtgp32-cuda.cu src/mtgp-1.1/mtgp32dc-param-11213.c
MTGP32_SRC = src/mtgp-1.1/mtgp32-fast.c src/mtgp-1.1/mtgp-util.cu src/mtgp-1.1/mtgp32-cuda.cu src/mtgp-1.1/mtgp32dc-param-11213.c
CUDALINK = -lcuda

NVCC_Dora=/home/santiago/cuda/bin/nvcc ${DEBUG} ${ARCH} -D__STDC_FORMAT_MACROS -L/home/santiago/cuda/fake-libs/ \
	-D__STDC_CONSTANT_MACROS ${VERB}
NVCC=nvcc ${DEBUG} ${ARCH} -D__STDC_FORMAT_MACROS -D__STDC_CONSTANT_MACROS ${VERB}

all: clu

dora: src/main.cu src/billionga.cu ${MTGP32_INC} 
	$(NVCC_Dora) -o bin/billionga \
		src/main.cu src/billionga.cu ${MTGP32_SRC} ${CUDALINK}
		
clu: src/main.cu src/billionga.cu ${MTGP32_INC} 
	$(NVCC) -o bin/billionga \
		src/main.cu src/billionga.cu ${MTGP32_SRC} ${CUDALINK}

