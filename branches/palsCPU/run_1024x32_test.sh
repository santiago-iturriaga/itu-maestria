DIMENSIONS="1024 32"
INSTANCES_PATH="instancias/1024x32"
SOLUTIONS_BASE_DIR="1024x32.test"
THREADS=8

VERIFICADOR="bin/verificador"
MINMIN_METRICS_PATH="list-heuristics/1024x32/MinMin"
RUSO_METRICS_PATH="pals-ruso/1024x32/palsRuso"

ALGORITHMS[0]="bin/pals_cpu"
ALGORITHMS_ID[0]=0
ALGORITHMS_OUTNAME[0]="pals.0"

ALGORITHMS[1]="bin/pals_cpu"
ALGORITHMS_ID[1]=1
ALGORITHMS_OUTNAME[1]="pals.1"

SCENARIOS[0]=10
SCENARIOS[1]=13
SCENARIOS[2]=19

WORKLOADS[0]=1
WORKLOADS[1]=10
WORKLOADS[2]=20
WORKLOADS[3]=30

#mkdir -p ${SOLUTIONS_DIR}

for a in {1..1}
do
    for s in {0..2}
    do
        for w in {0..3}
        do
            SOLUTIONS_DIR="${SOLUTIONS_BASE_DIR}/s${SCENARIOS[s]}.w${WORKLOADS[w]}"
            mkdir -p ${SOLUTIONS_DIR}
        
            OUT="${SOLUTIONS_DIR}/${ALGORITHMS_OUTNAME[a]}.s${SCENARIOS[s]}.w${WORKLOADS[w]}"
            rm ${OUT}.*
        
            RAND=0 #$RANDOM
            EXEC="${ALGORITHMS[a]} ${INSTANCES_PATH}/scenario.${SCENARIOS[s]} ${INSTANCES_PATH}/workload.${WORKLOADS[w]} ${DIMENSIONS} ${ALGORITHMS_ID[a]} ${THREADS} ${RAND}"
            echo ${EXEC}
            time (${EXEC} >> ${OUT}.sols 2> ${OUT}.info) 2> ${OUT}.time
        
            cat ${OUT}.time
        
            EXEC_VERIF="${VERIFICADOR} ${INSTANCES_PATH}/scenario.${SCENARIOS[s]} ${INSTANCES_PATH}/workload.${WORKLOADS[w]} ${OUT}.sols ${DIMENSIONS}"
            echo ${EXEC_VERIF}
            ${EXEC_VERIF} > ${OUT}.metrics
        
            echo "set term postscript" > ${OUT}.plot
            echo "set output '${OUT}.ps'" >> ${OUT}.plot
            echo "plot '${OUT}.metrics' using 1:2 title '${OUT}.metrics', '${MINMIN_METRICS_PATH}.s${SCENARIOS[s]}.w${WORKLOADS[w]}.metrics' using 1:2 title 'MinMin', '${RUSO_METRICS_PATH}.s${SCENARIOS[s]}.w${WORKLOADS[w]}.metrics' using 1:2 title 'Ruso'" >> ${OUT}.plot
            echo "set term png" >> ${OUT}.plot
            echo "set output '${OUT}.png'" >> ${OUT}.plot
            echo "replot" >> ${OUT}.plot
            gnuplot ${OUT}.plot

            echo "set term postscript" > ${OUT}.2.plot
            echo "set output '${OUT}.2.ps'" >> ${OUT}.2.plot
            echo "plot '${OUT}.metrics' using 1:2 title '${OUT}.metrics'" >> ${OUT}.2.plot
            echo "set term png" >> ${OUT}.2.plot
            echo "set output '${OUT}.2.png'" >> ${OUT}.2.plot
            echo "replot" >> ${OUT}.2.plot
            gnuplot ${OUT}.2.plot
        done
    done
done
