DIMENSIONS="1024 32"
INSTANCES_PATH="instancias/1024x32.ME"
SOLUTIONS_BASE_DIR="1024x32.speed"
ITERATIONS=30
PALS_ITERATIONS=6000000
PALS_TIMEOUT=60
POPULATION_SIZE=34

VERIFICADOR="bin/verificador"

ALGORITHMS[0]="bin/pals_cpu"
ALGORITHMS_ID[0]=1
ALGORITHMS_OUTNAME[0]="pals-aga"

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

for THREADS in {1..24}
do
    for a in {0..0}
    do
        for s in {0..0}
        do
            for w in {0..0}
            do
                for (( i=0; i<ITERATIONS; i++ ))
                do       
                    SOLUTIONS_DIR="${SOLUTIONS_BASE_DIR}.${THREADS}/scenario.${SCENARIOS[s]}.workload.${WORKLOADS[w]}.${i}"
                    mkdir -p ${SOLUTIONS_DIR}
                
                    echo ${SOLUTIONS_DIR}
                
                    OUT="${SOLUTIONS_DIR}/${ALGORITHMS_OUTNAME[a]}.scenario.${SCENARIOS[s]}.workload.${WORKLOADS[w]}"
                    rm ${OUT}.*
                
                    RAND=$RANDOM
                    EXEC="${ALGORITHMS[a]} ${INSTANCES_PATH}/scenario.${SCENARIOS[s]} ${INSTANCES_PATH}/workload.${WORKLOADS[w]} ${DIMENSIONS} ${ALGORITHMS_ID[a]} ${THREADS} ${RAND} ${PALS_TIMEOUT} ${PALS_ITERATIONS} ${POPULATION_SIZE}"
                    echo ${EXEC}
                    time (${EXEC} >> ${OUT}.sols 2> ${OUT}.info) 2> ${OUT}.time
                
                    cat ${OUT}.time
                
                    EXEC_VERIF="${VERIFICADOR} ${INSTANCES_PATH}/scenario.${SCENARIOS[s]} ${INSTANCES_PATH}/workload.${WORKLOADS[w]} ${OUT}.sols ${DIMENSIONS}"
                    echo ${EXEC_VERIF}
                    ${EXEC_VERIF} > ${OUT}.metrics
                done
                
                TODAS_LAS_SOLUCIONES="${SOLUTIONS_BASE_DIR}/scenario.${SCENARIOS[s]}.workload.${WORKLOADS[w]}"
                cat ${SOLUTIONS_BASE_DIR}/scenario.${SCENARIOS[s]}.workload.${WORKLOADS[w]}.*/*.metrics > ${TODAS_LAS_SOLUCIONES}.sols
                bin/fp_2obj ${TODAS_LAS_SOLUCIONES}.sols
                mv FP.out ${TODAS_LAS_SOLUCIONES}.fp
            done
        done
    done
done
