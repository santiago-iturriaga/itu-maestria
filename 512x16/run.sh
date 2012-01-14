BASE_PATH=".."
INSTANCE_PATH="/home/sergion/LUX/instancias/512x16"
SOL_PATH="solutions"
DIM_SIZE="512 16"
THREAD_COUNT="9"

mkdir -p ${SOL_PATH}

for INSTANCE in $(ls ${INSTANCE_PATH})
do
    echo ">>> Procesando ${INSTANCE}"

	echo "... minmin"
	time (${BASE_PATH}/bin/pals ${INSTANCE_PATH}/${INSTANCE} ${DIM_SIZE} 4 ${THREAD_COUNT} 0 > ${SOL_PATH}/${INSTANCE}.minmin.sol) 2> ${SOL_PATH}/${INSTANCE}.minmin.time
	${BASE_PATH}/bin/verificador ${INSTANCE_PATH}/${INSTANCE} ${SOL_PATH}/${INSTANCE}.minmin.sol ${DIM_SIZE} > ${SOL_PATH}/${INSTANCE}.minmin.makespan
	
	echo "... mct"
	time (${BASE_PATH}/bin/pals ${INSTANCE_PATH}/${INSTANCE} ${DIM_SIZE} 5 ${THREAD_COUNT} 0 > ${SOL_PATH}/${INSTANCE}.mct.sol) 2> ${SOL_PATH}/${INSTANCE}.mct.time
	${BASE_PATH}/bin/verificador ${INSTANCE_PATH}/${INSTANCE} ${SOL_PATH}/${INSTANCE}.mct.sol ${DIM_SIZE} > ${SOL_PATH}/${INSTANCE}.mct.makespan
		
	for (( j = 0; j < 15; j++ ))
	do
		SEED=$RANDOM
		echo "... pals CPU ${j} (seed: ${SEED})"

		#echo "time (${BASE_PATH}/bin/pals_cpu ${INSTANCE_PATH}/${INSTANCE} ${DIM_SIZE} 2 ${THREAD_COUNT} ${SEED} 1> ${SOL_PATH}/${INSTANCE}.palsCPU.${j}.sol 2> ${SOL_PATH}/${INSTANCE}.palsCPU.${j}.info) 2> ${SOL_PATH}/${INSTANCE}.palsCPU.${j}.time"

		time (${BASE_PATH}/bin/pals_cpu ${INSTANCE_PATH}/${INSTANCE} ${DIM_SIZE} 3 ${THREAD_COUNT} ${SEED} 1> ${SOL_PATH}/${INSTANCE}.palsCPU.${j}.sol 2> ${SOL_PATH}/${INSTANCE}.palsCPU.${j}.info) 2> ${SOL_PATH}/${INSTANCE}.palsCPU.${j}.time
		${BASE_PATH}/bin/verificador ${INSTANCE_PATH}/${INSTANCE} ${SOL_PATH}/${INSTANCE}.palsCPU.${j}.sol ${DIM_SIZE} > ${SOL_PATH}/${INSTANCE}.palsCPU.${j}.makespan
	done
done



