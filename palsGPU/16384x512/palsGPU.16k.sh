export LD_LIBRARY_PATH=/home/clusterusers/siturriaga/cuda/lib64:/home/clusterusers/siturriaga/cuda/lib

INSTANCE="/home/clusterusers/siturriaga/instances/Bernabe/palsGPU/etc_c_16384x512_hihi_"
BASE_PATH="/home/clusterusers/siturriaga/itu-maestria/trunk/palsGPU"
DIMENSION="16384 512"
DIMENSION_X="16384x512"

mkdir -p ${BASE_PATH}/16384x512/solutions

TIMEOUT=30
#TIMEOUT=120
#TIMEOUT=300
#TIMEOUT=900

TARGET_M=0
#TARGET_M=1901

MAX_ITER=20000
#MAX_ITER=1048576

for (( i=1; i<21; i++ ))
do
    set -x
    
    RAND=$RANDOM
    echo "Random ${RAND}"

    echo "=== PALS+MCT ==============================================="
    NAME="pals+mct"
    ID_1=2
    ID_2=5
    THREADS=1
    
    time (${BASE_PATH}/bin/pals ${INSTANCE}${i}.dat ${DIMENSION} ${ID_1} ${RAND} 0 ${TIMEOUT} ${TARGET_M} ${ID_2} ${THREADS} ${MAX_ITER} \
        1> ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.sol) \
        2> ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.time

    ${BASE_PATH}/bin/verificador ${INSTANCE}${i}.dat ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.sol ${DIMENSION} \
        > ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.makespan

    echo "=== PALS+pMINMIN 12 ==============================================="
    NAME="pals+pminmin+12"
    ID_1=2
    ID_2=3
    THREADS=12
    
    time (${BASE_PATH}/bin/pals ${INSTANCE}${i}.dat ${DIMENSION} ${ID_1} ${RAND} 0 ${TIMEOUT} ${TARGET_M} ${ID_2} ${THREADS} ${MAX_ITER} \
        1> ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.sol) \
        2> ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.time

    ${BASE_PATH}/bin/verificador ${INSTANCE}${i}.dat ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.sol ${DIMENSION} \
        > ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.makespan
                
    set +x
done
