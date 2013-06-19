#set +x

#export LD_LIBRARY_PATH=/home/clusterusers/siturriaga/cuda/lib64:/home/clusterusers/siturriaga/cuda/lib
export LD_LIBRARY_PATH=:/usr/local/cuda/lib64

INSTANCE="/home/clusterusers/siturriaga/instances/Bernabe/palsGPU/etc_c_8192x256_hihi_"
BASE_PATH="/home/clusterusers/siturriaga/itu-maestria/trunk/palsGPU"
DIMENSION="8192 256"
DIMENSION_X="8192x256"

hostname

mkdir -p ${BASE_PATH}/8192x256/solutions

ITER=30

TIMEOUT=30
#TIMEOUT=120
#TIMEOUT=300
#TIMEOUT=10

TARGET_M_ARRAY=(1845 1889 1894 1890 1859 1863 1897 1874 1871 1865 1840 1867 1895  1884 1851 1846 1874 1862  1892 1869)

#MAX_ITER=20000
#MAX_ITER=1048576
MAX_ITER=1073741824

for (( p=0; p<ITER; p++ ))
do
    for (( i=1; i<21; i++ ))
    do
        TARGET_M=0
        #TARGET_M=${TARGET_M_ARRAY[i-1]}
        
        RAND=$RANDOM
        echo "Random ${RAND}"

        set -x

        echo "=== PALS+gMinMin ==============================================="
        NAME="pals+gminmin"
        ID_1=2
        ID_2=6
        THREADS=1
        
        time (${BASE_PATH}/bin/pals ${INSTANCE}${i}.dat ${DIMENSION} ${ID_1} ${RAND} 0 ${TIMEOUT} ${TARGET_M} ${ID_2} ${THREADS} ${MAX_ITER} \
            ../mauro-sol/etc_c_${DIMENSION_X}_hihi-${i}.sol \
            1> ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.${p}.sol) \
            2> ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.${p}.time

        ${BASE_PATH}/bin/verificador ${INSTANCE}${i}.dat ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.${p}.sol ${DIMENSION} \
            > ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.${p}.makespan

        echo "=== PALS+MCT ==============================================="
        NAME="pals+mct"
        ID_1=2
        ID_2=5
        THREADS=1
        
        #time (${BASE_PATH}/bin/pals ${INSTANCE}${i}.dat ${DIMENSION} ${ID_1} ${RAND} 0 ${TIMEOUT} ${TARGET_M} ${ID_2} ${THREADS} ${MAX_ITER} \
        #    1> ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.${p}.sol) \
        #    2> ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.${p}.time

        #${BASE_PATH}/bin/verificador ${INSTANCE}${i}.dat ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.${p}.sol ${DIMENSION} \
        #    > ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.${p}.makespan

        echo "=== PALS+pMINMIN 12 ==============================================="
        NAME="pals+pminmin+12"
        ID_1=2
        ID_2=3
        THREADS=12
        
        #time (${BASE_PATH}/bin/pals ${INSTANCE}${i}.dat ${DIMENSION} ${ID_1} ${RAND} 0 ${TIMEOUT} ${TARGET_M} ${ID_2} ${THREADS} ${MAX_ITER} \
        #    1> ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.${p}.sol) \
        #    2> ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.${p}.time

        #${BASE_PATH}/bin/verificador ${INSTANCE}${i}.dat ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.${p}.sol ${DIMENSION} \
        #    > ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.${p}.makespan
                    
        set +x
    done
done
