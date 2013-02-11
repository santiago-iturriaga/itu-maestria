set -x

PRNG_PATH="prng"
mkdir -p ${PRNG_PATH}

DIMENSION_1="1024 32"
DIMENSION_2="1024x32"

ITER=30

PALS_TIMEOUT=512
PALS_ITER=6000000
PALS_POPULATION=34

THREADS[0]=1
THREADS[1]=2
THREADS[2]=4
THREADS[3]=6
THREADS[4]=8
THREADS[5]=10
THREADS[6]=12
THREADS[7]=14
THREADS[8]=16
THREADS[9]=18
THREADS[10]=20
THREADS[11]=22
THREADS[12]=24

INSTANCE="instancias/${DIMENSION_2}.ME/scenario.0 instancias/${DIMENSION_2}.ME/workload.A.u_c_hihi ${DIMENSION_1}"

for (( t=0; t<13; t++ ))
do
    for (( i=0; i<${ITER}; i++ ))
    do   
        time (bin/me-mls-adhoc-mct-mt ${INSTANCE} 1 ${THREADS[t]} 0 ${PALS_TIMEOUT} ${PALS_ITER} ${PALS_POPULATION} 1> ${PRNG_PATH}/aga.mt.${THREADS[t]}.${i}.sols 2> ${PRNG_PATH}/aga.mt.${THREADS[t]}.${i}.info) 2> ${PRNG_PATH}/aga.mt.${THREADS[t]}.${i}.time
        time (bin/me-mls-adhoc-mct-randr ${INSTANCE} 1 ${THREADS[t]} 0 ${PALS_TIMEOUT} ${PALS_ITER} ${PALS_POPULATION} 1> ${PRNG_PATH}/aga.randr.${THREADS[t]}.${i}.sols 2> ${PRNG_PATH}/aga.randr.${THREADS[t]}.${i}.info) 2> ${PRNG_PATH}/aga.randr.${THREADS[t]}.${i}.time
        time (bin/me-mls-adhoc-mct-drand48r ${INSTANCE} 1 ${THREADS[t]} 0 ${PALS_TIMEOUT} ${PALS_ITER} ${PALS_POPULATION} 1> ${PRNG_PATH}/aga.drand48r.${THREADS[t]}.${i}.sols 2> ${PRNG_PATH}/aga.drand48r.${THREADS[t]}.${i}.info) 2> ${PRNG_PATH}/aga.drand48r.${THREADS[t]}.${i}.time

        time (bin/me-mls-aga-mct-mt ${INSTANCE} 1 ${THREADS[t]} 0 ${PALS_TIMEOUT} ${PALS_ITER} ${PALS_POPULATION} 1> ${PRNG_PATH}/adhoc.mt.${THREADS[t]}.${i}.sols 2> ${PRNG_PATH}/adhoc.mt.${THREADS[t]}.${i}.info) 2> ${PRNG_PATH}/adhoc.mt.${THREADS[t]}.${i}.time
        time (bin/me-mls-aga-mct-randr ${INSTANCE} 1 ${THREADS[t]} 0 ${PALS_TIMEOUT} ${PALS_ITER} ${PALS_POPULATION} 1> ${PRNG_PATH}/adhoc.randr.${THREADS[t]}.${i}.sols 2> ${PRNG_PATH}/adhoc.randr.${THREADS[t]}.${i}.info) 2> ${PRNG_PATH}/adhoc.randr.${THREADS[t]}.${i}.time
        time (bin/me-mls-aga-mct-drand48r ${INSTANCE} 1 ${THREADS[t]} 0 ${PALS_TIMEOUT} ${PALS_ITER} ${PALS_POPULATION} 1> ${PRNG_PATH}/adhoc.drand48r.${THREADS[t]}.${i}.sols 2> ${PRNG_PATH}/adhoc.drand48r.${THREADS[t]}.${i}.info) 2> ${PRNG_PATH}/adhoc.drand48r.${THREADS[t]}.${i}.time
    done
done
