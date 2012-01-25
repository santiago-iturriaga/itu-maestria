DIMENSIONS="512 16"
INSTANCES_PATH="512x16"
SOLUTIONS_DIR="512x16.old"

SCENARIOS[0]=10
SCENARIOS[1]=13
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
			OUT="${SOLUTIONS_DIR}/s${SCENARIOS[s]}.w${WORKLOADS[w]}"
			rm ${OUT}.*
			
			echo "scenario.${SCENARIOS[s]} ${INSTANCES_PATH}/workload.${WORKLOADS[w]}"
			
			python sw2old_simple.py ${INSTANCES_PATH}/scenario.${SCENARIOS[s]} ${INSTANCES_PATH}/workload.${WORKLOADS[w]} ${DIMENSIONS} > ${OUT}
		done
	done
done
