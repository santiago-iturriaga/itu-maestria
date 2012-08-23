DIMENSIONS="512 16"
BASE_PATH=$(pwd)
ME_INSTANCES_PATH="512x16.ME"
SOLUTIONS_DIR="512x16.ME.old"

mkdir -p ${SOLUTIONS_DIR}
rm ${SOLUTIONS_DIR}/*.instance

cd ${ME_INSTANCES_PATH}
SCENARIOS=$(ls scenario.*)
WORKLOADS=$(ls workload.*)
cd ${BASE_PATH}

for s in ${SCENARIOS}
do
    for w in ${WORKLOADS}
    do
        echo ">>> ${s} ${w}"
        
        EXEC="python sw2old_simple.py ${ME_INSTANCES_PATH}/${s} ${ME_INSTANCES_PATH}/${w} ${DIMENSIONS}"
        echo ${EXEC}
        ${EXEC} > ${SOLUTIONS_DIR}/${s}.${w}
    done
done
