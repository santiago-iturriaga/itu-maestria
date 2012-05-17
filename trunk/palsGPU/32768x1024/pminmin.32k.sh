export LD_LIBRARY_PATH=/home/clusterusers/siturriaga/cuda/lib64:/home/clusterusers/siturriaga/cuda/lib

INSTANCE="/home/clusterusers/siturriaga/instances/Bernabe/palsGPU/etc_c_32768x1024_hihi_"
BASE_PATH="/home/clusterusers/siturriaga/itu-maestria/trunk/palsGPU"
DIMENSION="32768 1024"

mkdir -p ${BASE_PATH}/32768x1024/solutions

for (( i=1; i<21; i++ ))
do
    RAND=$RANDOM
    echo "Random ${RAND}"

    echo "time (${BASE_PATH}/bin/pals ${INSTANCE}${i}.dat ${DIMENSION} 3 ${RAND} 0 120 1> ${BASE_PATH}/32768x1024/solutions/${i}.pminmin.sol) 2> ${BASE_PATH}/32768x1024/solutions/${i}.pminmin.time"

    time (${BASE_PATH}/bin/pals ${INSTANCE}${i}.dat ${DIMENSION} 3 ${RAND} 0 120 \
        1> ${BASE_PATH}/32768x1024/solutions/${i}.pminmin.sol) \
        2> ${BASE_PATH}/32768x1024/solutions/${i}.pminmin.time

    echo "${BASE_PATH}/bin/verificador ${INSTANCE}${i}.dat ${BASE_PATH}/32768x1024/solutions/${i}.pminmin.sol ${DIMENSION} > ${BASE_PATH}/32768x1024/solutions/${i}.pminmin.makespan"

    ${BASE_PATH}/bin/verificador ${INSTANCE}${i}.dat ${BASE_PATH}/32768x1024/solutions/${i}.pminmin.sol ${DIMENSION} \
        > ${BASE_PATH}/32768x1024/solutions/${i}.pminmin.makespan
done
