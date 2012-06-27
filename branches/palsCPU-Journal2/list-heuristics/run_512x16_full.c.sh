DIMENSIONS="512 16"
INSTANCES_PATH="../instancias/512x16.ME.c"
SOLUTIONS_DIR="512x16.c"

ALGORITHMS[0]="MinMin"
ALGORITHMS[1]="MinMIN"
ALGORITHMS[2]="MINMin"
ALGORITHMS[3]="MINMIN"

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

mkdir -p ${SOLUTIONS_DIR}
rm ${SOLUTIONS_DIR}/*.metrics

for a in {0..3}
do
    for s in {0..19}
    do
        for w in {0..23}
        do
            for i in {0..2}
            do
                EXEC="./${ALGORITHMS[a]} ${INSTANCES_PATH}/workload.${WORKLOADS[w]}_${i} ${INSTANCES_PATH}/scenario.${s} ${DIMENSIONS}"
                echo ${EXEC}
                ${EXEC} > ${SOLUTIONS_DIR}/${ALGORITHMS[a]}.scenario.${s}.workload.${WORKLOADS[w]}_${i}.metrics
                cat ${SOLUTIONS_DIR}/${ALGORITHMS[a]}.scenario.${s}.workload.${WORKLOADS[w]}_${i}.metrics
            done
        done
    done
done
