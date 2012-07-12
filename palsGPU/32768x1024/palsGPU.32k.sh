export LD_LIBRARY_PATH=/home/clusterusers/siturriaga/cuda/lib64:/home/clusterusers/siturriaga/cuda/lib

INSTANCE="/home/clusterusers/siturriaga/instances/Bernabe/palsGPU/etc_c_32768x1024_hihi_"
BASE_PATH="/home/clusterusers/siturriaga/itu-maestria/trunk/palsGPU"
DIMENSION="32768 1024"
DIMENSION_X="32768x1024"

mkdir -p ${BASE_PATH}/32768x1024/solutions

ITER=30

TIMEOUT=30
#TIMEOUT=120
#TIMEOUT=300
#TIMEOUT=900

TARGET_M_ARRAY=(1996 1979 1980 1982 1971 1973 1991 1991 1994 1997 1975 1974 1978 1988 1972 1979 1991 1986 1975 1991)

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

        echo "=== PALS+MCT ==============================================="
        NAME="pals+mct"
        ID_1=2
        ID_2=5
        THREADS=1
        
        time (${BASE_PATH}/bin/pals ${INSTANCE}${i}.dat ${DIMENSION} ${ID_1} ${RAND} 0 ${TIMEOUT} ${TARGET_M} ${ID_2} ${THREADS} ${MAX_ITER} \
            1> ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.${p}.sol) \
            2> ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.${p}.time

        ${BASE_PATH}/bin/verificador ${INSTANCE}${i}.dat ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.${p}.sol ${DIMENSION} \
            > ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.${p}.makespan

        echo "=== PALS+pMINMIN 12 ==============================================="
        NAME="pals+pminmin+12"
        ID_1=2
        ID_2=3
        THREADS=12
        
        time (${BASE_PATH}/bin/pals ${INSTANCE}${i}.dat ${DIMENSION} ${ID_1} ${RAND} 0 ${TIMEOUT} ${TARGET_M} ${ID_2} ${THREADS} ${MAX_ITER} \
            1> ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.${p}.sol) \
            2> ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.${p}.time

        ${BASE_PATH}/bin/verificador ${INSTANCE}${i}.dat ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.${p}.sol ${DIMENSION} \
            > ${BASE_PATH}/${DIMENSION_X}/solutions/${i}.${NAME}.${p}.makespan
                    
        set +x
    done
done
