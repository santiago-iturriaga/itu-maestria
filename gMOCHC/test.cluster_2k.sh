set -x

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

for s in {0..19}
do
    for w in {0..23}
    do

        #MinMin
        bin/gmochc_cpu ~/instancias/2048x64.ME/scenario.${s} ~/instancias/2048x64.ME/workload.${WORKLOADS[w]} \
            2048 64 1 1 0 1 1 1>minmin_${s}_${WORKLOADS[w]}_2k.sol 2>minmin_${s}_${WORKLOADS[w]}_2k.log

        #cMOCHC/islands
        SEED=0
        #THREADS=2
        THREADS=24
        TIMEOUT=1
        #ITERATIONS=20
        #ITERATIONS=1250
        ITERATIONS=2000
        time(bin/gmochc_cpu ~/instancias/2048x64.ME/scenario.${s} ~/instancias/2048x64.ME/workload.${WORKLOADS[w]} \
            2048 64 3 ${THREADS} ${SEED} ${TIMEOUT} ${ITERATIONS} 1>chc__${s}_${WORKLOADS[w]}_2k.sol 2>chc__${s}_${WORKLOADS[w]}_2k.log)

    done
done
