INSTANCE=$1
PATH="/home/clusterusers/siturriaga/itu-maestria/trunk/palsGPU"
DIMENSION="32768 1024"

time (${PATH}/bin/minmin ${PATH}/32768x1024/instances/${INSTANCE} ${DIMENSION} \
    1> ${PATH}/32768x1024/solutions/${INSTANCE}.minmin.sol) \
        2> ${PATH}/32768x1024/solutions/${INSTANCE}.minmin.time

${PATH}/bin/verificador ${PATH}/32768x1024/instances/${INSTANCE} ${PATH}/32768x1024/solutions/${INSTANCE}.minmin.sol ${DIMENSION} > ${PATH}/32768x1024/solutions/${INSTANCE}.minmin.makespan
