DIMENSIONS="2048 64"
INSTANCES_PATH="instancias/2048x64.ME"
SOLUTIONS_BASE_DIR="2048x64.test"
THREADS=8
ITERATIONS=15

VERIFICADOR="bin/verificador"
MINMIN_METRICS_PATH="list-heuristics/2048x64/MinMin"
RUSO_METRICS_PATH="pals-ruso/2048x64/pals-ruso"

ALGORITHMS[0]="bin/pals_cpu"
ALGORITHMS_ID[0]=1
ALGORITHMS_OUTNAME[0]="pals-1"

SCENARIOS[0]=10
SCENARIOS[1]=13
SCENARIOS[2]=16
SCENARIOS[3]=19

WORKLOADS[0]="A.u_c_hihi"
WORKLOADS[1]="A.u_c_lolo"
WORKLOADS[2]="A.u_i_hihi"
WORKLOADS[3]="A.u_i_lolo"
WORKLOADS[4]="A.u_s_hihi"
WORKLOADS[5]="A.u_s_lolo"

#mkdir -p ${SOLUTIONS_DIR}
#rm ${SOLUTIONS_DIR}/*.metrics
#rm ${SOLUTIONS_DIR}/*.sols

for a in {0..0}
do
    for s in {0..0}
    do
        for w in {0..1}
        do
            for (( i=0; i<ITERATIONS; i++ ))
            do       
                SOLUTIONS_DIR="${SOLUTIONS_BASE_DIR}/scenario.${SCENARIOS[s]}.workload.${WORKLOADS[w]}.${i}"
                mkdir -p ${SOLUTIONS_DIR}
            
                echo ${SOLUTIONS_DIR}
            
                OUT="${SOLUTIONS_DIR}/${ALGORITHMS_OUTNAME[a]}.scenario.${SCENARIOS[s]}.workload.${WORKLOADS[w]}"
                rm ${OUT}.*
            
                RAND=$RANDOM
                EXEC="${ALGORITHMS[a]} ${INSTANCES_PATH}/scenario.${SCENARIOS[s]} ${INSTANCES_PATH}/workload.${WORKLOADS[w]} ${DIMENSIONS} ${ALGORITHMS_ID[a]} ${THREADS} ${RAND}"
                echo ${EXEC}
                time (${EXEC} >> ${OUT}.sols 2> ${OUT}.info) 2> ${OUT}.time
            
                cat ${OUT}.time
            
                EXEC_VERIF="${VERIFICADOR} ${INSTANCES_PATH}/scenario.${SCENARIOS[s]} ${INSTANCES_PATH}/workload.${WORKLOADS[w]} ${OUT}.sols ${DIMENSIONS}"
                echo ${EXEC_VERIF}
                ${EXEC_VERIF} > ${OUT}.metrics
            
                echo "set term postscript" > ${OUT}.plot
                echo "set output '${OUT}.ps'" >> ${OUT}.plot
                echo "plot '${OUT}.metrics' using 1:2 title 'PALS2obj', '${MINMIN_METRICS_PATH}.scenario.${SCENARIOS[s]}.workload.${WORKLOADS[w]}.metrics' using 1:2 title 'MinMin', '${RUSO_METRICS_PATH}.scenario.${SCENARIOS[s]}.workload.${WORKLOADS[w]}.metrics' using 1:2 title 'Ruso'" >> ${OUT}.plot
                echo "set term png" >> ${OUT}.plot
                echo "set output '${OUT}.png'" >> ${OUT}.plot
                echo "replot" >> ${OUT}.plot
                gnuplot ${OUT}.plot

                echo "set term postscript" > ${OUT}.2.plot
                echo "set output '${OUT}.2.ps'" >> ${OUT}.2.plot
                echo "plot '${OUT}.metrics' using 1:2 title 'PALS2obj'" >> ${OUT}.2.plot
                echo "set term png" >> ${OUT}.2.plot
                echo "set output '${OUT}.2.png'" >> ${OUT}.2.plot
                echo "replot" >> ${OUT}.2.plot
                gnuplot ${OUT}.2.plot
            done
            
            TODAS_LAS_SOLUCIONES="${SOLUTIONS_BASE_DIR}/scenario.${SCENARIOS[s]}.workload.${WORKLOADS[w]}"
            cat ${SOLUTIONS_BASE_DIR}/scenario.${SCENARIOS[s]}.workload.${WORKLOADS[w]}.*/*.metrics > ${TODAS_LAS_SOLUCIONES}.sols
            bin/fp_2obj ${TODAS_LAS_SOLUCIONES}.sols
            mv FP.out ${TODAS_LAS_SOLUCIONES}.fp
            
            echo "set term postscript" > ${TODAS_LAS_SOLUCIONES}.plot
            echo "set output '${TODAS_LAS_SOLUCIONES}.ps'" >> ${TODAS_LAS_SOLUCIONES}.plot
            echo "plot '${TODAS_LAS_SOLUCIONES}.fp' using 1:2 title 'PALS2obj', '${MINMIN_METRICS_PATH}.scenario.${SCENARIOS[s]}.workload.${WORKLOADS[w]}.metrics' using 1:2 title 'MinMin', '${RUSO_METRICS_PATH}.scenario.${SCENARIOS[s]}.workload.${WORKLOADS[w]}.metrics' using 1:2 title 'Ruso'" >> ${TODAS_LAS_SOLUCIONES}.plot
            echo "set term png" >> ${TODAS_LAS_SOLUCIONES}.plot
            echo "set output '${TODAS_LAS_SOLUCIONES}.png'" >> ${TODAS_LAS_SOLUCIONES}.plot
            echo "replot" >> ${TODAS_LAS_SOLUCIONES}.plot
            gnuplot ${TODAS_LAS_SOLUCIONES}.plot
        done
    done
done
