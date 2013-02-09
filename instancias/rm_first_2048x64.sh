DIMENSIONS="2048 64"
BASE_PATH=$(pwd)
ME_INSTANCES_PATH="2048x64.M.dim"
SOLUTIONS_DIR="2048x64.M"

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
