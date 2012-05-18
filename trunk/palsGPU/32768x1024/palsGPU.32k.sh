export LD_LIBRARY_PATH=/home/clusterusers/siturriaga/cuda/lib64:/home/clusterusers/siturriaga/cuda/lib

INSTANCE="/home/clusterusers/siturriaga/instances/Bernabe/palsGPU/etc_c_32768x1024_hihi_"
BASE_PATH="/home/clusterusers/siturriaga/itu-maestria/trunk/palsGPU"
DIMENSION="32768 1024"
DIMENSION_X="32768x1024"

mkdir -p ${BASE_PATH}/32768x1024/solutions

TIMEOUT=30
#TIMEOUT=120
#TIMEOUT=300
#TIMEOUT=900

TARGET_M=0
#TARGET_M=1971

for (( i=1; i<21; i++ ))
do
    RAND=$RANDOM
    echo "Random ${RAND}"

    echo "=== PALS+pMINMIN 12 ==============================================="
    NAME="pals+pminmin+12"
    ID=3
    THREADS=12
    
    echo "time (${BASE_PATH}/bin/pals ${INSTANCE}${i}.dat ${DIMENSION} 2 ${RAND} 0 ${TIMEOUT} ${TARGET_M} ${ID} ${THREADS} 1> ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.sol) 2> ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.time"

    time (${BASE_PATH}/bin/pals ${INSTANCE}${i}.dat ${DIMENSION} 2 ${RAND} 0 ${TIMEOUT} ${TARGET_M} ${ID} ${THREADS} \
        1> ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.sol) \
        2> ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.time

    echo "${BASE_PATH}/bin/verificador ${INSTANCE}${i}.dat ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.sol ${DIMENSION} > ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.makespan"

    ${BASE_PATH}/bin/verificador ${INSTANCE}${i}.dat ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.sol ${DIMENSION} \
        > ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.makespan

done
