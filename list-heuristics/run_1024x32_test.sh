DIMENSIONS="1024 32"
INSTANCES_PATH="../instancias/1024x32.ME"
SOLUTIONS_DIR="1024x32"

ALGORITHMS[0]="MinMin"
ALGORITHMS[1]="MinMIN"
ALGORITHMS[2]="MINMin"
ALGORITHMS[3]="MINMIN"

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

mkdir -p ${SOLUTIONS_DIR}
rm ${SOLUTIONS_DIR}/*.metrics

for a in {0..3}
do
    for s in {0..3}
    do
        for w in {0..5}
        do               
            EXEC="./${ALGORITHMS[a]} ${INSTANCES_PATH}/workload.${WORKLOADS[w]} ${INSTANCES_PATH}/scenario.${SCENARIOS[s]} ${DIMENSIONS}"
            echo ${EXEC}
            ${EXEC} > ${SOLUTIONS_DIR}/${ALGORITHMS[a]}.scenario.${SCENARIOS[s]}.workload.${WORKLOADS[w]}.metrics
            cat ${SOLUTIONS_DIR}/${ALGORITHMS[a]}.scenario.${SCENARIOS[s]}.workload.${WORKLOADS[w]}.metrics
        done
    done
done
