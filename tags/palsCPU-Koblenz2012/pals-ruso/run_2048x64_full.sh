DIMENSIONS="2048 64"
ARGS="12 6 2 500000 8 8"
INSTANCES_PATH="../instancias/2048x64.ME.old"
INSTANCES_NEW_PATH="../instancias/2048x64.ME"
SOLUTIONS_DIR="2048x64"
VERIFICADOR="../bin/verificador"

SCENARIOS[0]=0
SCENARIOS[1]=3
SCENARIOS[2]=6
SCENARIOS[3]=9
SCENARIOS[4]=10
SCENARIOS[5]=11
SCENARIOS[6]=13
SCENARIOS[7]=14
SCENARIOS[8]=16
SCENARIOS[9]=17
SCENARIOS[10]=19

WORKLOADS[0]="A.u_c_hihi"
WORKLOADS[1]="A.u_c_hilo"
WORKLOADS[2]="A.u_c_lohi"
WORKLOADS[3]="A.u_c_lolo"
WORKLOADS[4]="A.u_i_hihi"
WORKLOADS[5]="A.u_i_hilo"
WORKLOADS[6]="A.u_i_lohi"
WORKLOADS[7]="A.u_i_lolo"
WORKLOADS[8]="A.u_s_hihi"
WORKLOADS[9]="A.u_s_hilo"
WORKLOADS[10]="A.u_s_lohi"
WORKLOADS[11]="A.u_s_lolo"
WORKLOADS[12]="B.u_c_hihi"
WORKLOADS[13]="B.u_c_hilo"
WORKLOADS[14]="B.u_c_lohi"
WORKLOADS[15]="B.u_c_lolo"
WORKLOADS[16]="B.u_i_hihi"
WORKLOADS[17]="B.u_i_hilo"
WORKLOADS[18]="B.u_i_lohi"
WORKLOADS[19]="B.u_i_lolo"
WORKLOADS[20]="B.u_s_hihi"
WORKLOADS[21]="B.u_s_hilo"
WORKLOADS[22]="B.u_s_lohi"
WORKLOADS[23]="B.u_s_lolo"

mkdir ${SOLUTIONS_DIR}
rm ${SOLUTIONS_DIR}/*.metrics
rm ${SOLUTIONS_DIR}/*.sols

for s in {0..10}
do
    for w in {0..23}
    do
        INSTANCE="scenario.${SCENARIOS[s]}.workload.${WORKLOADS[w]}"
        OUT="${SOLUTIONS_DIR}/pals-ruso.${INSTANCE}"
        
        EXEC="./palsRuso.2048 ${INSTANCES_PATH}/${INSTANCE} ${ARGS}"
        echo ${EXEC}
        ${EXEC} > ${OUT}.sol
            
        EXEC_VERIF="${VERIFICADOR} ${INSTANCES_NEW_PATH}/scenario.${SCENARIOS[s]} ${INSTANCES_NEW_PATH}/workload.${WORKLOADS[w]} ${OUT}.sol ${DIMENSIONS}"
        echo ${EXEC_VERIF}
        ${EXEC_VERIF} > ${OUT}.metrics
    done
done
