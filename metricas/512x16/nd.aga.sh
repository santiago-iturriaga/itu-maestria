set -x

SOLUTIONS_BASE_PATH="../../512x16.24_10s"
METRICS_PATH="metrics-aga"
ITERATIONS=30
ALGORITHM="pals-aga"

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

for s in {0..10}
do
    for w in {0..23}
    do
        mkdir -p ${METRICS_PATH}

        for (( i=0; i<ITERATIONS; i++ ))
        do       
            SOLUTIONS_DIR="${SOLUTIONS_BASE_PATH}/scenario.${SCENARIOS[s]}.workload.${WORKLOADS[w]}.${i}"

            IN="${SOLUTIONS_DIR}/${ALGORITHM}.scenario.${SCENARIOS[s]}.workload.${WORKLOADS[w]}.metrics"
            OUT="${METRICS_PATH}/${ALGORITHM}.scenario.${SCENARIOS[s]}.workload.${WORKLOADS[w]}.${i}.nd"
        
            wc -l ${IN} > ${OUT}
        done
    done
done
