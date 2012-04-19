# Makefile

#DEBUG = -DDEBUG -g -O0

#ARCH = -arch=compute_11
#ARCH = -arch=compute_13
#VERB = -Xptxas=-v

NVCC = /home/santiago/cuda/bin/nvcc ${DEBUG} ${ARCH} -D__STDC_FORMAT_MACROS -I./ -L/home/santiago/cuda/fake-libs/ \
	-D__STDC_CONSTANT_MACROS ${VERB}

MTGP32_INC = mtgp32-fast.h mtgp32-fast.c mtgp-util.cuh mtgp-util.cu
MTGP32_SRC = mtgp32-fast.c mtgp-util.cu
CUDALINK = -lcuda

#.SUFFIXES:	.cu .cubin .ptx .bin

samples:sample32-cuda

sample32-cuda: mtgp32-cuda.cu ${MTGP32_INC} mtgp32dc-param-11213.c
	${NVCC} -o $@ mtgp32-cuda.cu mtgp32dc-param-11213.c \
	${MTGP32_SRC} ${CUDALINK}

#.cu.ptx:
#	${NVCC} -ptx $<

#.cu.cubin:
#	${NVCC} -cubin $<

#.cu.bin:
#	${NVCC} -o $@ $< ${CUDALINK}

clean:
	rm -f *.o *~ *.cubin *.ptx
