set -x

DIMENSIONS="1024 32"
INSTANCES_PATH="instancias/1024x32.ME"
SOLUTIONS_BASE_DIR="1024x32.24_10s.m2.t"

THREADS=24
ITERATIONS=30

PALS_ITERATIONS=250000000
PALS_TIMEOUT=10
PALS_POP_SIZE=34

VERIFICADOR="bin/verificador"
ALGORITHMS[0]="bin/pals_cpu"
ALGORITHMS_ID[0]=1
ALGORITHMS_OUTNAME[0]="pals-aga"

for a in {0..0}
do
    for s in {0..10}
    do
        for w in {10..14}
        do
            for (( i=0; i<ITERATIONS; i++ ))
            do       
                SOLUTIONS_DIR="${SOLUTIONS_BASE_DIR}/scenario.${s}.workload.${w}.${i}"
                mkdir -p ${SOLUTIONS_DIR}
            
                echo ${SOLUTIONS_DIR}
            
                OUT="${SOLUTIONS_DIR}/${ALGORITHMS_OUTNAME[a]}.scenario.${s}.workload.${w}"
                rm ${OUT}.*
            
                RAND=$RANDOM
                EXEC="${ALGORITHMS[a]} ${INSTANCES_PATH}/scenario.${s} ${INSTANCES_PATH}/workload.${w} ${DIMENSIONS} ${ALGORITHMS_ID[a]} ${THREADS} ${RAND} ${PALS_TIMEOUT} ${PALS_ITERATIONS} ${PALS_POP_SIZE}"
                echo ${EXEC}
                time (${EXEC} >> ${OUT}.sols 2> ${OUT}.info) 2> ${OUT}.time
            
                cat ${OUT}.time
            
                EXEC_VERIF="${VERIFICADOR} ${INSTANCES_PATH}/scenario.${s} ${INSTANCES_PATH}/workload.${w} ${OUT}.sols ${DIMENSIONS}"
                echo ${EXEC_VERIF}
                ${EXEC_VERIF} > ${OUT}.metrics
            done
            
            TODAS_LAS_SOLUCIONES="${SOLUTIONS_BASE_DIR}/scenario.${s}.workload.${w}"
            cat ${SOLUTIONS_BASE_DIR}/scenario.${SCENARIOS[s]}.workload.${WORKLOADS[w]}.*/*.metrics > ${TODAS_LAS_SOLUCIONES}.sols
            bin/fp_2obj ${TODAS_LAS_SOLUCIONES}.sols
            mv FP.out ${TODAS_LAS_SOLUCIONES}.fp
        done
        
        for w in {20..24}
        do
            for (( i=0; i<ITERATIONS; i++ ))
            do       
                SOLUTIONS_DIR="${SOLUTIONS_BASE_DIR}/scenario.${s}.workload.${w}.${i}"
                mkdir -p ${SOLUTIONS_DIR}
            
                echo ${SOLUTIONS_DIR}
            
                OUT="${SOLUTIONS_DIR}/${ALGORITHMS_OUTNAME[a]}.scenario.${s}.workload.${w}"
                rm ${OUT}.*
            
                RAND=$RANDOM
                EXEC="${ALGORITHMS[a]} ${INSTANCES_PATH}/scenario.${s} ${INSTANCES_PATH}/workload.${w} ${DIMENSIONS} ${ALGORITHMS_ID[a]} ${THREADS} ${RAND} ${PALS_TIMEOUT} ${PALS_ITERATIONS} ${PALS_POP_SIZE}"
                echo ${EXEC}
                time (${EXEC} >> ${OUT}.sols 2> ${OUT}.info) 2> ${OUT}.time
            
                cat ${OUT}.time
            
                EXEC_VERIF="${VERIFICADOR} ${INSTANCES_PATH}/scenario.${s} ${INSTANCES_PATH}/workload.${w} ${OUT}.sols ${DIMENSIONS}"
                echo ${EXEC_VERIF}
                ${EXEC_VERIF} > ${OUT}.metrics
            done
            
            TODAS_LAS_SOLUCIONES="${SOLUTIONS_BASE_DIR}/scenario.${s}.workload.${w}"
            cat ${SOLUTIONS_BASE_DIR}/scenario.${SCENARIOS[s]}.workload.${WORKLOADS[w]}.*/*.metrics > ${TODAS_LAS_SOLUCIONES}.sols
            bin/fp_2obj ${TODAS_LAS_SOLUCIONES}.sols
            mv FP.out ${TODAS_LAS_SOLUCIONES}.fp
        done
        
        for w in {30..34}
        do
            for (( i=0; i<ITERATIONS; i++ ))
            do       
                SOLUTIONS_DIR="${SOLUTIONS_BASE_DIR}/scenario.${s}.workload.${w}.${i}"
                mkdir -p ${SOLUTIONS_DIR}
            
                echo ${SOLUTIONS_DIR}
            
                OUT="${SOLUTIONS_DIR}/${ALGORITHMS_OUTNAME[a]}.scenario.${s}.workload.${w}"
                rm ${OUT}.*
            
                RAND=$RANDOM
                EXEC="${ALGORITHMS[a]} ${INSTANCES_PATH}/scenario.${s} ${INSTANCES_PATH}/workload.${w} ${DIMENSIONS} ${ALGORITHMS_ID[a]} ${THREADS} ${RAND} ${PALS_TIMEOUT} ${PALS_ITERATIONS} ${PALS_POP_SIZE}"
                echo ${EXEC}
                time (${EXEC} >> ${OUT}.sols 2> ${OUT}.info) 2> ${OUT}.time
            
                cat ${OUT}.time
            
                EXEC_VERIF="${VERIFICADOR} ${INSTANCES_PATH}/scenario.${s} ${INSTANCES_PATH}/workload.${w} ${OUT}.sols ${DIMENSIONS}"
                echo ${EXEC_VERIF}
                ${EXEC_VERIF} > ${OUT}.metrics
            done
            
            TODAS_LAS_SOLUCIONES="${SOLUTIONS_BASE_DIR}/scenario.${s}.workload.${w}"
            cat ${SOLUTIONS_BASE_DIR}/scenario.${SCENARIOS[s]}.workload.${WORKLOADS[w]}.*/*.metrics > ${TODAS_LAS_SOLUCIONES}.sols
            bin/fp_2obj ${TODAS_LAS_SOLUCIONES}.sols
            mv FP.out ${TODAS_LAS_SOLUCIONES}.fp
        done
    done
done
