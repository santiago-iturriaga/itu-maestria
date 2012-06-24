DIMENSIONS="1024 32"
INSTANCES_PATH="../instancias/1024x32"
SOLUTIONS_DIR="1024x32"

ALGORITHMS[0]="MinMin"
ALGORITHMS[1]="MinMIN"
ALGORITHMS[2]="MINMin"
ALGORITHMS[3]="MINMIN"

mkdir -p ${SOLUTIONS_DIR}
rm ${SOLUTIONS_DIR}/*.metrics

for a in {0..3}
do
    for s in {0..19}
    do
        for w in {0..39}
        do               
            EXEC="./${ALGORITHMS[a]} ${INSTANCES_PATH}/workload.${w} ${INSTANCES_PATH}/scenario.${s} ${DIMENSIONS}"
            echo ${EXEC}
            ${EXEC} > ${SOLUTIONS_DIR}/${ALGORITHMS[a]}.scenario.${s}.workload.${w}.metrics
            cat ${SOLUTIONS_DIR}/${ALGORITHMS[a]}.scenario.${s}.workload.${w}.metrics
        done
    done
done
