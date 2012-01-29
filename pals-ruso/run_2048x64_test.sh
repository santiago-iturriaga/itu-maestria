DIMENSIONS="2048 64"
ARGS="12 6 2 500000 8 8"
INSTANCES_PATH="../instancias/2048x64.ME.old"
INSTANCES_NEW_PATH="../instancias/2048x64"
SOLUTIONS_DIR="2048x64"
VERIFICADOR="../bin/verificador"

SCENARIOS[0]=10
SCENARIOS[1]=13
SCENARIOS[2]=19

WORKLOADS[0]=1
WORKLOADS[1]=10
WORKLOADS[2]=20
WORKLOADS[3]=30

mkdir ${SOLUTIONS_DIR}

for s in {0..2}
do
    for w in {0..3}
    do
        rm ${SOLUTIONS_DIR}/palsRuso.s${SCENARIOS[s]}.w${WORKLOADS[w]}.*
        
        ./palsRuso.1024 ${INSTANCES_PATH}/s${SCENARIOS[s]}.w${WORKLOADS[w]} ${ARGS} > ${SOLUTIONS_DIR}/palsRuso.s${SCENARIOS[s]}.w${WORKLOADS[w]}.sols
            
        EXEC_VERIF="${VERIFICADOR} ${INSTANCES_NEW_PATH}/scenario.${SCENARIOS[s]} ${INSTANCES_NEW_PATH}/workload.${WORKLOADS[w]} ${SOLUTIONS_DIR}/palsRuso.s${SCENARIOS[s]}.w${WORKLOADS[w]}.sols ${DIMENSIONS}"
        echo ${EXEC_VERIF}
        ${EXEC_VERIF} > ${SOLUTIONS_DIR}/palsRuso.s${SCENARIOS[s]}.w${WORKLOADS[w]}.metrics
    done
done
