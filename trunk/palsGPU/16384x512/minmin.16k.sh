INSTANCE="/home/clusterusers/siturriaga/instances/Bernabe/palsGPU/etc_c_16384x512_hihi_"
BASE_PATH="/home/clusterusers/siturriaga/itu-maestria/trunk/palsGPU"
DIMENSION="16384 512"

mkdir -p ${BASE_PATH}/16384x512/solutions

for (( i=1; i<21; i++ ))
do
    echo "time (${BASE_PATH}/bin/minmin ${INSTANCE}${i}.dat ${DIMENSION} 1> ${BASE_PATH}/16384x512/solutions/${i}.minmin.sol) 2> ${BASE_PATH}/16384x512/solutions/${i}.minmin.time"

    time (${BASE_PATH}/bin/minmin ${INSTANCE}${i}.dat ${DIMENSION} \
	1> ${BASE_PATH}/16384x512/solutions/${i}.minmin.sol) \
	2> ${BASE_PATH}/16384x512/solutions/${i}.minmin.time

    echo "${BASE_PATH}/bin/verificador ${INSTANCE}${i}.dat ${BASE_PATH}/16384x512/solutions/${i}.minmin.sol ${DIMENSION} > ${BASE_PATH}/16384x512/solutions/${i}.minmin.makespan"

    ${BASE_PATH}/bin/verificador ${INSTANCE}${i}.dat ${BASE_PATH}/16384x512/solutions/${i}.minmin.sol ${DIMENSION} \
	> ${BASE_PATH}/16384x512/solutions/${i}.minmin.makespan
done
