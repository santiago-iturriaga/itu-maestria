DIMENSIONS="512 16"
INSTANCES_PATH="instancias/512x16"
SOLUTIONS_BASE_DIR="512x16.test"
THREADS=16
VERIFICADOR="bin/verificador"
ITERATIONS=15

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

mkdir -p ${SOLUTIONS_DIR}

for a in {0..1}
do
	for s in {0..2}
	do
		for w in {0..3}
		do
            for (( i=0; i < ${ITERATIONS}; i++ ))
            do
                SOLUTIONS_DIR="${SOLUTIONS_BASE_DIR}/s${SCENARIOS[s]}.w${WORKLOADS[w]}.${i}"
                mkdir -p ${SOLUTIONS_DIR}
                
                OUT="${SOLUTIONS_DIR}/${ALGORITHMS_OUTNAME[a]}.s${SCENARIOS[s]}.w${WORKLOADS[w]}"
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
                echo "plot '${OUT}.metrics' using 1:2 title '${OUT}'" >> ${OUT}.plot
                echo "set term png" >> ${OUT}.plot
                echo "set output '${OUT}.png'" >> ${OUT}.plot
                echo "replot" >> ${OUT}.plot
                gnuplot ${OUT}.plot
            done
		done
	done
done
