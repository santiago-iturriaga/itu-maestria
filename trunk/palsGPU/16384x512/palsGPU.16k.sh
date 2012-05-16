export LD_LIBRARY_PATH=/home/clusterusers/siturriaga/cuda/lib64:/home/clusterusers/siturriaga/cuda/lib

INSTANCE="/home/clusterusers/siturriaga/instances/Bernabe/palsGPU/etc_c_16384x512_hihi_"
BASE_PATH="/home/clusterusers/siturriaga/itu-maestria/trunk/palsGPU"
DIMENSION="16384 512"

mkdir -p ${BASE_PATH}/16384x512/solutions

for (( i=1; i<21; i++ ))
do
    RAND=$RANDOM
    echo "Random ${RAND}"

    echo "time (${BASE_PATH}/bin/pals ${INSTANCE}${i}.dat ${DIMENSION} 2 ${RAND} 0 90 1> ${BASE_PATH}/16384x512/solutions/${i}.pals.sol) 2> ${BASE_PATH}/16384x512/solutions/${i}.pals.time"

    time (${BASE_PATH}/bin/pals ${INSTANCE}${i}.dat ${DIMENSION} 2 ${RAND} 0 90 \
	1> ${BASE_PATH}/16384x512/solutions/${i}.pals.sol) \
	2> ${BASE_PATH}/16384x512/solutions/${i}.pals.time

    echo "${BASE_PATH}/bin/verificador ${INSTANCE}${i}.dat ${BASE_PATH}/16384x512/solutions/${i}.pals.sol ${DIMENSION} > ${BASE_PATH}/16384x512/solutions/${i}.pals.makespan"

    ${BASE_PATH}/bin/verificador ${INSTANCE}${i}.dat ${BASE_PATH}/16384x512/solutions/${i}.pals.sol ${DIMENSION} \
	> ${BASE_PATH}/16384x512/solutions/${i}.pals.makespan
done

