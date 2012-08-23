DIMENSIONS="512 16"
INSTANCES_PATH="../instancias/512x16"
SOLUTIONS_DIR="512x16"

ALGORITHMS[0]="MinMin"
ALGORITHMS[1]="MinMIN"
ALGORITHMS[2]="MINMin"
ALGORITHMS[3]="MINMIN"

SCENARIOS[0]=10
SCENARIOS[1]=13
SCENARIOS[2]=19

WORKLOADS[0]=1
WORKLOADS[1]=10
WORKLOADS[2]=20
WORKLOADS[3]=30

mkdir ${SOLUTIONS_DIR}

for a in {0..3}
do
    for s in {0..2}
    do
        for w in {0..3}
        do
            rm ${SOLUTIONS_DIR}/${ALGORITHMS[a]}.s${SCENARIOS[s]}.w${WORKLOADS[w]}.metrics
            
            ./${ALGORITHMS[a]} ${INSTANCES_PATH}/workload.${WORKLOADS[w]} ${INSTANCES_PATH}/scenario.${SCENARIOS[s]} ${DIMENSIONS} \
                >> ${SOLUTIONS_DIR}/${ALGORITHMS[a]}.s${SCENARIOS[s]}.w${WORKLOADS[w]}.metrics
        done
    done
done
