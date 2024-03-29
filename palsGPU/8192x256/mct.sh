export LD_LIBRARY_PATH=/home/clusterusers/siturriaga/cuda/lib64:/home/clusterusers/siturriaga/cuda/lib

INSTANCE="/home/clusterusers/siturriaga/instances/Bernabe/palsGPU/etc_c_8192x256_hihi_"
BASE_PATH="/home/clusterusers/siturriaga/itu-maestria/trunk/palsGPU"
DIMENSION="8192 256"
DIMENSION_X="8192x256"

mkdir -p ${BASE_PATH}/8192x256/solutions

TIMEOUT=30
#TIMEOUT=120
#TIMEOUT=300
#TIMEOUT=900

TARGET_M=0
#TARGET_M=1840

for (( i=1; i<21; i++ ))
do
    RAND=$RANDOM
    echo "Random ${RAND}"

    echo "=== MCT ==============================================="
    NAME="mct"
    ID=5
    THREADS=1
    
    echo "time (${BASE_PATH}/bin/pals ${INSTANCE}${i}.dat ${DIMENSION} 5 ${RAND} 0 ${TIMEOUT} ${TARGET_M} ${ID} ${THREADS} 1> ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.sol) 2> ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.time"

    time (${BASE_PATH}/bin/pals ${INSTANCE}${i}.dat ${DIMENSION} 5 ${RAND} 0 ${TIMEOUT} ${TARGET_M} ${ID} ${THREADS} \
        1> ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.sol) \
        2> ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.time

    echo "${BASE_PATH}/bin/verificador ${INSTANCE}${i}.dat ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.sol ${DIMENSION} > ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.makespan"

    ${BASE_PATH}/bin/verificador ${INSTANCE}${i}.dat ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.sol ${DIMENSION} \
        > ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.makespan
done

