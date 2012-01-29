DIMENSIONS="2048 64"
ARGS="12 6 2 500000 8 8"
INSTANCES_PATH="../instancias/2048x64.ME.old"
INSTANCES_NEW_PATH="../instancias/2048x64.ME"
SOLUTIONS_DIR="2048x64"
VERIFICADOR="../bin/verificador"
BASE_PATH=$(pwd)

cd ${INSTANCES_PATH}
INSTANCES=$(ls scenario.*.workload.*)
cd ${BASE_PATH}

mkdir -p ${SOLUTIONS_DIR}
rm ${SOLUTIONS_DIR}/*.metrics
rm ${SOLUTIONS_DIR}/*.sols

for instance in ${INSTANCES}
do
    OUT="${SOLUTIONS_DIR}/pals-ruso.${instance}"

    EXEC="./palsRuso.2048 ${INSTANCES_PATH}/${instance} ${ARGS}"
    echo ${EXEC}
    
    ${EXEC} > ${OUT}.sol
    
    NAME_SPLITTED=$(echo ${instance} | tr "." "\n")
    POS=0
    for part in ${NAME_SPLITTED}
    do
        NAME_ARRAY[POS]=${part}
        POS=$((POS+1))       
    done

    scenario_num=${NAME_ARRAY[1]}
    workload_prefix=${NAME_ARRAY[3]}
    workload=${NAME_ARRAY[4]}
    
    EXEC_VERIF="${VERIFICADOR} ${INSTANCES_NEW_PATH}/scenario.${scenario_num} ${INSTANCES_NEW_PATH}/workload.${workload_prefix}.${workload} ${OUT}.sol ${DIMENSIONS}"
    echo ${EXEC_VERIF}
    ${EXEC_VERIF} > ${OUT}.metrics
done
