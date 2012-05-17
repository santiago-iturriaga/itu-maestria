export LD_LIBRARY_PATH=/home/clusterusers/siturriaga/cuda/lib64:/home/clusterusers/siturriaga/cuda/lib

INSTANCE="/home/clusterusers/siturriaga/instances/Bernabe/palsGPU/etc_c_32768x1024_hihi_"
BASE_PATH="/home/clusterusers/siturriaga/itu-maestria/trunk/palsGPU"
DIMENSION="32768 1024"

mkdir -p ${BASE_PATH}/32768x1024/solutions

#TIMEOUT=120
TIMEOUT=300
#TIMEOUT=900
TARGET_M=0
#TARGET_M=1971

for (( i=1; i<21; i++ ))
do
    RAND=$RANDOM
    echo "Random ${RAND}"

    echo "=== MCT ===================================================="
    echo "time (${BASE_PATH}/bin/pals ${INSTANCE}${i}.dat ${DIMENSION} 5 ${RAND} 0 120 1> ${BASE_PATH}/32768x1024/solutions/${i}.mct.sol) 2> ${BASE_PATH}/32768x1024/solutions/${i}.mct.time"

    time (${BASE_PATH}/bin/pals ${INSTANCE}${i}.dat ${DIMENSION} 5 ${RAND} 0 120 \
        1> ${BASE_PATH}/32768x1024/solutions/${i}.mct.sol) \
        2> ${BASE_PATH}/32768x1024/solutions/${i}.mct.time

    echo "${BASE_PATH}/bin/verificador ${INSTANCE}${i}.dat ${BASE_PATH}/32768x1024/solutions/${i}.mct.sol ${DIMENSION} > ${BASE_PATH}/32768x1024/solutions/${i}.mct.makespan"

    ${BASE_PATH}/bin/verificador ${INSTANCE}${i}.dat ${BASE_PATH}/32768x1024/solutions/${i}.mct.sol ${DIMENSION} \
        > ${BASE_PATH}/32768x1024/solutions/${i}.mct.makespan

    echo "=== PALS+MCT ==============================================="
    echo "time (${BASE_PATH}/bin/pals ${INSTANCE}${i}.dat ${DIMENSION} 2 ${RAND} 0 ${TIMEOUT} ${TARGET_M} 5 1> ${BASE_PATH}/32768x1024/solutions/${i}.pals+mct.sol) 2> ${BASE_PATH}/32768x1024/solutions/${i}.pals+mct.time"

    time (${BASE_PATH}/bin/pals ${INSTANCE}${i}.dat ${DIMENSION} 2 ${RAND} 0 ${TIMEOUT} ${TARGET_M} 5 \
        1> ${BASE_PATH}/32768x1024/solutions/${i}.pals+mct.sol) \
        2> ${BASE_PATH}/32768x1024/solutions/${i}.pals+mct.time

    echo "${BASE_PATH}/bin/verificador ${INSTANCE}${i}.dat ${BASE_PATH}/32768x1024/solutions/${i}.pals+mct.sol ${DIMENSION} > ${BASE_PATH}/32768x1024/solutions/${i}.pals+mct.makespan"

    ${BASE_PATH}/bin/verificador ${INSTANCE}${i}.dat ${BASE_PATH}/32768x1024/solutions/${i}.pals+mct.sol ${DIMENSION} \
        > ${BASE_PATH}/32768x1024/solutions/${i}.pals+mct.makespan
        
    echo "=== PALS+pMINMIN ==============================================="
    echo "time (${BASE_PATH}/bin/pals ${INSTANCE}${i}.dat ${DIMENSION} 2 ${RAND} 0 ${TIMEOUT} ${TARGET_M} 3 1> ${BASE_PATH}/32768x1024/solutions/${i}.pals+pminmin.sol) 2> ${BASE_PATH}/32768x1024/solutions/${i}.pals+pminmin.time"

    time (${BASE_PATH}/bin/pals ${INSTANCE}${i}.dat ${DIMENSION} 2 ${RAND} 0 ${TIMEOUT} ${TARGET_M} 3 \
        1> ${BASE_PATH}/32768x1024/solutions/${i}.pals+pminmin.sol) \
        2> ${BASE_PATH}/32768x1024/solutions/${i}.pals+pminmin.time

    echo "${BASE_PATH}/bin/verificador ${INSTANCE}${i}.dat ${BASE_PATH}/32768x1024/solutions/${i}.pals+pminmin.sol ${DIMENSION} > ${BASE_PATH}/32768x1024/solutions/${i}.pals+pminmin.makespan"

    ${BASE_PATH}/bin/verificador ${INSTANCE}${i}.dat ${BASE_PATH}/32768x1024/solutions/${i}.pals+pminmin.sol ${DIMENSION} \
        > ${BASE_PATH}/32768x1024/solutions/${i}.pals+pminmin.makespan
done
