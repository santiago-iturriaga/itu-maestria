DIMENSIONS="1024 32"
BASE_PATH=$(pwd)
ME_INSTANCES_PATH="1024x32.M.dim"
SOLUTIONS_DIR="1024x32.M"

mkdir -p ${SOLUTIONS_DIR}

cd ${ME_INSTANCES_PATH}
SCENARIOS=$(ls *)
cd ${BASE_PATH}

for s in ${SCENARIOS}
do
	echo ">>> ${s}"
        
	EXEC="python remove_first_line.py ${ME_INSTANCES_PATH}/${s}"
	echo ${EXEC}
	${EXEC} > ${SOLUTIONS_DIR}/${s}
done
