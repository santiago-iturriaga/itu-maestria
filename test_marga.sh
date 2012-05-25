set -x
rm -rf 512x16.test
rm bin/pals_cpu
make

echo "MinMin"
bin/pals_cpu instancias/512x16.ME/scenario.0 instancias/512x16.ME/workload.B.u_c_hihi 512 16 2 1 1 1 1 1
echo "MCT"
bin/pals_cpu instancias/512x16.ME/scenario.0 instancias/512x16.ME/workload.B.u_c_hihi 512 16 3 1 1 1 1 1

DIMENSIONS="512 16"
INSTANCES_PATH="instancias/512x16.ME"
SOLUTIONS_BASE_DIR="512x16.test"

#THREADS=1
THREADS=2
#THREADS=4

ITERATIONS=1
PALS_ITERATIONS=100000000
PALS_TIMEOUT=30
#PALS_TIMEOUT=90

PALS_POP_SIZE=6
#PALS_POP_SIZE=16
#PALS_POP_SIZE=300

VERIFICADOR="bin/verificador"
MINMIN_METRICS_PATH="list-heuristics/512x16/MinMin"
RUSO_METRICS_PATH="pals-ruso/512x16/pals-ruso"

ALGORITHMS[0]="bin/pals_cpu"
ALGORITHMS_ID[0]=1
ALGORITHMS_OUTNAME[0]="pals-1"

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

#mkdir -p ${SOLUTIONS_DIR}
#rm ${SOLUTIONS_DIR}/*.metrics
#rm ${SOLUTIONS_DIR}/*.sols

for a in {0..0}
do
    for s in {0..0}
    do
        for w in {12..12}
        do
            for (( i=0; i<ITERATIONS; i++ ))
            do       
                SOLUTIONS_DIR="${SOLUTIONS_BASE_DIR}/scenario.${SCENARIOS[s]}.workload.${WORKLOADS[w]}.${i}"
                mkdir -p ${SOLUTIONS_DIR}
            
                echo ${SOLUTIONS_DIR}
            
                OUT="${SOLUTIONS_DIR}/${ALGORITHMS_OUTNAME[a]}.scenario.${SCENARIOS[s]}.workload.${WORKLOADS[w]}"
                rm ${OUT}.*
            
                RAND=1 #$RANDOM
                EXEC="${ALGORITHMS[a]} ${INSTANCES_PATH}/scenario.${SCENARIOS[s]} ${INSTANCES_PATH}/workload.${WORKLOADS[w]} ${DIMENSIONS} ${ALGORITHMS_ID[a]} ${THREADS} ${RAND} ${PALS_TIMEOUT} ${PALS_ITERATIONS} ${PALS_POP_SIZE}"
                echo ${EXEC}
                #time (${EXEC} >> ${OUT}.sols 2> ${OUT}.info) 2> ${OUT}.time
                time (${EXEC})
            
                cat ${OUT}.time
            
                EXEC_VERIF="${VERIFICADOR} ${INSTANCES_PATH}/scenario.${SCENARIOS[s]} ${INSTANCES_PATH}/workload.${WORKLOADS[w]} ${OUT}.sols ${DIMENSIONS}"
                echo ${EXEC_VERIF}
                ${EXEC_VERIF} > ${OUT}.metrics
            done
        done
    done
done
