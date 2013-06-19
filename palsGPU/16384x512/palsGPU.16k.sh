#export LD_LIBRARY_PATH=/home/clusterusers/siturriaga/cuda/lib64:/home/clusterusers/siturriaga/cuda/lib
export LD_LIBRARY_PATH=:/usr/local/cuda/lib64

INSTANCE="/home/clusterusers/siturriaga/instances/Bernabe/palsGPU/etc_c_16384x512_hihi_"
BASE_PATH="/home/clusterusers/siturriaga/itu-maestria/trunk/palsGPU"
DIMENSION="16384 512"
DIMENSION_X="16384x512"

hostname

mkdir -p ${BASE_PATH}/16384x512/solutions

ITER=30

TIMEOUT=30
#TIMEOUT=120
#TIMEOUT=300
#TIMEOUT=900

TARGET_M_ARRAY=(1934 1940 1949 1922 1904 1901 1945 1903 1937 1935 1937 1916 1911 1927 1944 1939 1933 1929 1910 1941)

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
