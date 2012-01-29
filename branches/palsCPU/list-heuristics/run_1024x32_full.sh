DIMENSIONS="1024 32"
INSTANCES_PATH="../instancias/1024x32.ME"
BASE_PATH=$(pwd)
SOLUTIONS_DIR="1024x32"

ALGORITHMS[0]="MinMin"
ALGORITHMS[1]="MinMIN"
ALGORITHMS[2]="MINMin"
ALGORITHMS[3]="MINMIN"

cd ${INSTANCES_PATH}
SCENARIOS=$(ls scenario.*)
WORKLOADS=$(ls workload.*)
cd ${BASE_PATH}

mkdir -p ${SOLUTIONS_DIR}
rm ${SOLUTIONS_DIR}/*.metrics

for a in {0..3}
do
    for s in ${SCENARIOS}
    do
        for w in ${WORKLOADS}
        do
            echo ">>>> ${s} ${w}"

            EXEC="./${ALGORITHMS[a]} ${INSTANCES_PATH}/${w} ${INSTANCES_PATH}/${s} ${DIMENSIONS}"
            echo ${EXEC}
            ${EXEC} > ${SOLUTIONS_DIR}/${ALGORITHMS[a]}.${s}.${w}.metrics
            cat ${SOLUTIONS_DIR}/${ALGORITHMS[a]}.${s}.${w}.metrics
        done
    done
done
