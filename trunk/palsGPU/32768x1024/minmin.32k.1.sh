INSTANCE="/home/clusterusers/siturriaga/instances/Bernabe/palsGPU/etc_c_32768x1024_hihi_"
BASE_PATH="/home/clusterusers/siturriaga/itu-maestria/trunk/palsGPU"
DIMENSION="32768 1024"

mkdir -p ${BASE_PATH}/32768x1024/solutions

for (( i=1; i<11; i++ ))
do
    echo "time (${BASE_PATH}/bin/minmin ${INSTANCE}${i}.dat ${DIMENSION} 1> ${BASE_PATH}/32768x1024/solutions/${i}.minmin.sol) 2> ${BASE_PATH}/32768x1024/solutions/${i}.minmin.time"

    time (${BASE_PATH}/bin/minmin ${INSTANCE}${i}.dat ${DIMENSION} \
        1> ${BASE_PATH}/32768x1024/solutions/${i}.minmin.sol) \
        2> ${BASE_PATH}/32768x1024/solutions/${i}.minmin.time

    echo "${BASE_PATH}/bin/verificador ${INSTANCE}${i}.dat ${BASE_PATH}/32768x1024/solutions/${i}.minmin.sol ${DIMENSION} > ${BASE_PATH}/32768x1024/solutions/${i}.minmin.makespan"

    ${BASE_PATH}/bin/verificador ${INSTANCE}${i}.dat ${BASE_PATH}/32768x1024/solutions/${i}.minmin.sol ${DIMENSION} \
        > ${BASE_PATH}/32768x1024/solutions/${i}.minmin.makespan
done
