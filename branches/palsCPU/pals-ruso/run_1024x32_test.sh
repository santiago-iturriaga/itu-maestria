DIMENSIONS="1024 32"
ARGS="12 6 2 500000 8 8"
INSTANCES_PATH="../instancias/1024x32.ME.old"
INSTANCES_NEW_PATH="../instancias/1024x32.ME"
SOLUTIONS_DIR="1024x32"
VERIFICADOR="../bin/verificador"

SCENARIOS[0]=10
SCENARIOS[1]=13
SCENARIOS[2]=16
SCENARIOS[3]=19

WORKLOADS[0]="A.u_c_hihi"
WORKLOADS[1]="A.u_c_lolo"
WORKLOADS[2]="A.u_i_hihi"
WORKLOADS[3]="A.u_i_lolo"
WORKLOADS[4]="A.u_s_hihi"
WORKLOADS[5]="A.u_s_lolo"

mkdir ${SOLUTIONS_DIR}
rm ${SOLUTIONS_DIR}/*.metrics
rm ${SOLUTIONS_DIR}/*.sols

for s in {0..3}
do
    for w in {0..5}
    do
        INSTANCE="scenario.${SCENARIOS[s]}.workload.${WORKLOADS[w]}"
        OUT="${SOLUTIONS_DIR}/pals-ruso.${INSTANCE}"
        
        EXEC="./palsRuso.1024 ${INSTANCES_PATH}/${INSTANCE} ${ARGS}"
        echo ${EXEC}
        ${EXEC} > ${OUT}.sol
            
        EXEC_VERIF="${VERIFICADOR} ${INSTANCES_NEW_PATH}/scenario.${SCENARIOS[s]} ${INSTANCES_NEW_PATH}/workload.${WORKLOADS[w]} ${OUT}.sol ${DIMENSIONS}"
        echo ${EXEC_VERIF}
        ${EXEC_VERIF} > ${OUT}.metrics
    done
done
