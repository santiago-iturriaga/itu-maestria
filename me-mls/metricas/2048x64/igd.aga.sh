set -x

cd /home/siturria/itu-maestria/branches/palsCPU-Journal2/metricas/2048x64

#CLASSPATH="/home/santiago/itu-maestria/svn/trunk/metricas_mo/java/bin"
CLASSPATH="/home/siturria/itu-maestria/trunk/metricas_mo/java/bin"
NUM_OBJ=2

SOLUTIONS_BASE_PATH="../../2048x64.24_10s"
FP_BASE_PATH="../../2048x64.fp"
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
WORKLOADS[4]="B.u_c_hihi"
WORKLOADS[5]="B.u_c_hilo"
WORKLOADS[6]="B.u_c_lohi"
WORKLOADS[7]="B.u_c_lolo"

#WORKLOADS[0]="A.u_c_hihi"
#WORKLOADS[1]="A.u_c_hilo"
#WORKLOADS[2]="A.u_c_lohi"
#WORKLOADS[3]="A.u_c_lolo"
#WORKLOADS[4]="A.u_i_hihi"
#WORKLOADS[5]="A.u_i_hilo"
#WORKLOADS[6]="A.u_i_lohi"
#WORKLOADS[7]="A.u_i_lolo"
#WORKLOADS[8]="A.u_s_hihi"
#WORKLOADS[9]="A.u_s_hilo"
#WORKLOADS[10]="A.u_s_lohi"
#WORKLOADS[11]="A.u_s_lolo"
#WORKLOADS[12]="B.u_c_hihi"
#WORKLOADS[13]="B.u_c_hilo"
#WORKLOADS[14]="B.u_c_lohi"
#WORKLOADS[15]="B.u_c_lolo"
#WORKLOADS[16]="B.u_i_hihi"
#WORKLOADS[17]="B.u_i_hilo"
#WORKLOADS[18]="B.u_i_lohi"
#WORKLOADS[19]="B.u_i_lolo"
#WORKLOADS[20]="B.u_s_hihi"
#WORKLOADS[21]="B.u_s_hilo"
#WORKLOADS[22]="B.u_s_lohi"
#WORKLOADS[23]="B.u_s_lolo"

for s in {0..10}
do
    for w in {0..7}
    do
        mkdir -p ${METRICS_PATH}
        FP="${FP_BASE_PATH}/scenario.${SCENARIOS[s]}.workload.${WORKLOADS[w]}.fp"

        for (( i=0; i<ITERATIONS; i++ ))
        do       
            SOLUTIONS_DIR="${SOLUTIONS_BASE_PATH}/scenario.${SCENARIOS[s]}.workload.${WORKLOADS[w]}.${i}"

            IN="${SOLUTIONS_DIR}/${ALGORITHM}.scenario.${SCENARIOS[s]}.workload.${WORKLOADS[w]}.metrics"
            OUT="${METRICS_PATH}/${ALGORITHM}.scenario.${SCENARIOS[s]}.workload.${WORKLOADS[w]}.${i}.igd"
            
            #java -classpath $CLASSPATH jmetal.qualityIndicator.InvertedGenerationalDistanceNonNormalized ${IN} ${FP} $NUM_OBJ > ${OUT}
            java -classpath $CLASSPATH jmetal.qualityIndicator.InvertedGenerationalDistance ${IN} ${FP} $NUM_OBJ > ${OUT}
        done
    done
done
