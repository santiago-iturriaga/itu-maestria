BASE_PATH=$(pwd)
SRC_PATH="32768x1024.src"
DEST_PATH="32768x1024.M"

mkdir -p ${DEST_PATH}

cd ${SRC_PATH}
SCENARIOS=$(ls *)
cd ${BASE_PATH}

for s in ${SCENARIOS}
do
	echo ">>> ${s}"
        
	EXEC="python remove_first_line.py ${SRC_PATH}/${s}"
	echo ${EXEC}
	${EXEC} > ${DEST_PATH}/${s}
done
