set -x

PRNG_PATH="prng"
mkdir -p ${PRNG_PATH}

DIMENSION_1="512 16"
DIMENSION_2="512x16"

ITER=5

PALS_TIMEOUT=512
PALS_ITER=6000000
PALS_POPULATION=18

THREADS[0]=1
THREADS[1]=2
THREADS[2]=4
THREADS[3]=8

INSTANCE="instancias/${DIMENSION_2}.ME/scenario.0 instancias/${DIMENSION_2}.ME/workload.A.u_c_hihi ${DIMENSION_1}"

for (( t=0; t<4; t++ ))
do
    for (( i=0; i<${ITER}; i++ ))
    do
        time (bin/pals_cpu_mt_prof ${INSTANCE} 1 ${THREADS[t]} 0 ${PALS_POPULATION} ${PALS_ITER} ${PALS_TIMEOUT} 1> ${PRNG_PATH}/gmon.mt.${THREADS[t]}.${i}.sols 2> ${PRNG_PATH}/gmon.mt.${THREADS[t]}.${i}.info) 2> ${PRNG_PATH}/gmon.mt.${THREADS[t]}.${i}.time
        gprof bin/pals_cpu_mt_prof gmon.out > ${PRNG_PATH}/gmon.mt.${THREADS[t]}.${i}.txt

        time (bin/pals_cpu_randr_prof ${INSTANCE} 1 ${THREADS[t]} 0 ${PALS_POPULATION} ${PALS_ITER} ${PALS_TIMEOUT} 1> ${PRNG_PATH}/gmon.randr.${THREADS[t]}.${i}.sols 2> ${PRNG_PATH}/gmon.randr.${THREADS[t]}.${i}.info) 2> ${PRNG_PATH}/gmon.randr.${THREADS[t]}.${i}.time
        gprof bin/pals_cpu_mt_prof gmon.out > ${PRNG_PATH}/gmon.randr.${THREADS[t]}.${i}.txt

        time (bin/pals_cpu_drand48r_prof ${INSTANCE} 1 ${THREADS[t]} 0 ${PALS_POPULATION} ${PALS_ITER} ${PALS_TIMEOUT} 1> ${PRNG_PATH}/gmon.drand48r.${THREADS[t]}.${i}.sols 2> ${PRNG_PATH}/gmon.drand48r.${THREADS[t]}.${i}.info) 2> ${PRNG_PATH}/gmon.drand48r.${THREADS[t]}.${i}.time
        gprof bin/pals_cpu_mt_prof gmon.out > ${PRNG_PATH}/gmon.drand48r.${THREADS[t]}.${i}.txt
    done
done
