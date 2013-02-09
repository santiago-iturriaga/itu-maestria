DIMENSIONS="512 16"
BASE=$(pwd)
ME_INSTANCES_PATH="512x16"
M_INSTANCES_PATH="512x16.M"
SOLUTIONS_DIR="512x16.ME"

mkdir -p ${SOLUTIONS_DIR}

i=0

cp ${ME_INSTANCES_PATH}/scenario.* ${SOLUTIONS_DIR}

cd ${M_INSTANCES_PATH}
files=$(ls *)
cd ${BASE}

for workload_file in ${files}
do
    echo "${workload_file} to workload.${i}"
    
    cp ${M_INSTANCES_PATH}/${workload_file} ${SOLUTIONS_DIR}/workload.${workload_file}
    
    #echo ${EXEC}
    #${EXEC}
    
    i=$((i+1))
done
