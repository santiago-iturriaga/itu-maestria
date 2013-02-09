DIMENSIONS="4096 128"
BASE_PATH=$(pwd)
SOLUTIONS_DIR="4096x128.M"

mkdir -p ${SOLUTIONS_DIR}

cd ${SOLUTIONS_DIR}
SCENARIOS=$(ls *)
cd ${BASE_PATH}

for s in ${SCENARIOS}
do
	echo ">>> ${s}"
        
    mv ${SOLUTIONS_DIR}/${s} ${SOLUTIONS_DIR}/${s}.dim
	EXEC="python remove_first_line.py ${SOLUTIONS_DIR}/${s}.dim"
	echo ${EXEC}
	${EXEC} > ${SOLUTIONS_DIR}/${s}
done
