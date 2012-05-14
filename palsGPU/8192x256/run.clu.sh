export LD_LIBRARY_PATH=/home/clusterusers/siturriaga/cuda/lib64:/home/clusterusers/siturriaga/cuda/lib

BASE_PATH=".."
INSTANCE_PATH="instances"
SOL_PATH="solutions"
DIM_SIZE="8192 256"
GPU_DEVICE="1"

mkdir -p ${SOL_PATH}

for INSTANCE in $(ls ${INSTANCE_PATH})
do
    echo ">>> Procesando ${INSTANCE}"

	#echo "... minmin"
	#time (${BASE_PATH}/bin/pals ${INSTANCE_PATH}/${INSTANCE} ${DIM_SIZE} 4 0 ${GPU_DEVICE} > ${SOL_PATH}/${INSTANCE}.minmin.sol) 2> ${SOL_PATH}/${INSTANCE}.minmin.time
	#${BASE_PATH}/bin/verificador ${INSTANCE_PATH}/${INSTANCE} ${SOL_PATH}/${INSTANCE}.minmin.sol ${DIM_SIZE} > ${SOL_PATH}/${INSTANCE}.minmin.makespan
	
	#echo "... mct"
	#time (${BASE_PATH}/bin/pals ${INSTANCE_PATH}/${INSTANCE} ${DIM_SIZE} 5 0 ${GPU_DEVICE} > ${SOL_PATH}/${INSTANCE}.mct.sol) 2> ${SOL_PATH}/${INSTANCE}.mct.time
	#${BASE_PATH}/bin/verificador ${INSTANCE_PATH}/${INSTANCE} ${SOL_PATH}/${INSTANCE}.mct.sol ${DIM_SIZE} > ${SOL_PATH}/${INSTANCE}.mct.makespan
		
	for (( j = 0; j < 1; j++ ))
	do
		SEED=$RANDOM
		echo "... pals GPU ${j} (seed: ${SEED})"

        echo "${BASE_PATH}/bin/pals ${INSTANCE_PATH}/${INSTANCE} ${DIM_SIZE} 2 ${SEED} ${GPU_DEVICE} 1> ${SOL_PATH}/${INSTANCE}.palsGPU.${j}.sol 2> ${SOL_PATH}/${INSTANCE}.palsGPU.${j}.info) 2> ${SOL_PATH}/${INSTANCE}.palsGPU.${j}.time"
		echo "${BASE_PATH}/bin/verificador ${INSTANCE_PATH}/${INSTANCE} ${SOL_PATH}/${INSTANCE}.palsGPU.${j}.sol ${DIM_SIZE} > ${SOL_PATH}/${INSTANCE}.palsGPU.${j}.makespan"

		time (${BASE_PATH}/bin/pals ${INSTANCE_PATH}/${INSTANCE} ${DIM_SIZE} 2 ${SEED} ${GPU_DEVICE} 1> ${SOL_PATH}/${INSTANCE}.palsGPU.${j}.sol 2> ${SOL_PATH}/${INSTANCE}.palsGPU.${j}.info) 2> ${SOL_PATH}/${INSTANCE}.palsGPU.${j}.time
		${BASE_PATH}/bin/verificador ${INSTANCE_PATH}/${INSTANCE} ${SOL_PATH}/${INSTANCE}.palsGPU.${j}.sol ${DIM_SIZE} > ${SOL_PATH}/${INSTANCE}.palsGPU.${j}.makespan
	done
done
