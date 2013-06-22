#export LD_LIBRARY_PATH=/home/clusterusers/siturriaga/cuda/lib64:/home/clusterusers/siturriaga/cuda/lib
export LD_LIBRARY_PATH=:/usr/local/cuda/lib64

INSTANCE="/home/clusterusers/siturriaga/instances/Bernabe/palsGPU/etc_c_8192x256_hihi_"
BASE_PATH="/home/clusterusers/siturriaga/itu-maestria/trunk/palsGPU"
DIMENSION="8192 256"
DIMENSION_X="8192x256"

hostname

mkdir -p ${BASE_PATH}/8192x256/solutions_shared

ITER=30

#TIMEOUT=30
#TIMEOUT=120
#TIMEOUT=300
TIMEOUT=900

#TARGET_M_ARRAY=(1845 1889 1894 1890 1859 1863 1897 1874 1871 1865 1840 1867 1895  1884 1851 1846 1874 1862  1892 1869)
TARGET_M_ARRAY=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)

#MAX_ITER=20000
MAX_ITER=150000
#MAX_ITER=1048576
#MAX_ITER=1073741824

for (( p=0; p<ITER; p++ ))
do
    for (( i=1; i<2; i++ ))
    do
        TARGET_M=${TARGET_M_ARRAY[i-1]}
        
        RAND=$RANDOM
        echo "Random ${RAND}"

        set -x

        echo "=== PALS+MCT SHARED ==============================================="
        NAME="pals+shared"
        ID_1=2
        ID_2=5
        THREADS=1
        
        time (${BASE_PATH}/pals.shared ${INSTANCE}${i}.dat ${DIMENSION} ${ID_1} ${RAND} 0 ${TIMEOUT} ${TARGET_M} ${ID_2} ${THREADS} ${MAX_ITER} \
            1> ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.${p}.sol) \
            2> ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.${p}.time

        ${BASE_PATH}/bin/verificador ${INSTANCE}${i}.dat ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.${p}.sol ${DIMENSION} \
            > ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.${p}.makespan

        echo "=== PALS+MCT NO SHARED ==============================================="
        NAME="pals+no-shared"
        ID_1=2
        ID_2=5
        THREADS=1
        
        time (${BASE_PATH}/pals.no-shared ${INSTANCE}${i}.dat ${DIMENSION} ${ID_1} ${RAND} 0 ${TIMEOUT} ${TARGET_M} ${ID_2} ${THREADS} ${MAX_ITER} \
            1> ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.${p}.sol) \
            2> ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.${p}.time

        ${BASE_PATH}/bin/verificador ${INSTANCE}${i}.dat ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.${p}.sol ${DIMENSION} \
            > ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.${p}.makespan

                    
        set +x
    done
done
