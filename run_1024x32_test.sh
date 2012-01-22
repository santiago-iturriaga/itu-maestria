DIMENSIONS="1024 32"
INSTANCES_PATH="instancias/1024x32"
SOLUTIONS_DIR="1024x32.test"
THREADS=3

ALGORITHMS[0]="bin/pals_cpu"
ALGORITHMS_OUTNAME[0]="pals.rand_r.a3"

SCENARIOS[0]=0
SCENARIOS[1]=10
SCENARIOS[2]=19

WORKLOADS[0]=1
WORKLOADS[1]=10
WORKLOADS[2]=20
WORKLOADS[3]=30

mkdir ${SOLUTIONS_DIR}

for a in {0..0}
do
	for s in {0..2}
	do
		for w in {0..3}
		do
			OUT="${SOLUTIONS_DIR}/${ALGORITHMS_OUTNAME[a]}.s${SCENARIOS[s]}.w${WORKLOADS[w]}"
			rm ${OUT}.*
			
			RAND=$RANDOM
			EXEC="${ALGORITHMS[a]} ${INSTANCES_PATH}/scenario.${SCENARIOS[s]} ${INSTANCES_PATH}/workload.${WORKLOADS[w]} ${DIMENSIONS} 3 ${THREADS} ${RAND}"
			
			echo ${EXEC}
			time (${EXEC} >> ${OUT}.sols 2> ${OUT}.err) 2> ${OUT}.time
		done
	done
done
