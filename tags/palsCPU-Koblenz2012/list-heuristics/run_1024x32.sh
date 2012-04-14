INSTANCES_PATH=../instancias/1024x32

make clean
make all

echo "MinMin"
./MinMin ${INSTANCES_PATH}/workload.30 ${INSTANCES_PATH}/scenario.19 1024 32
echo "MinMIN"
./MinMIN ${INSTANCES_PATH}/workload.30 ${INSTANCES_PATH}/scenario.19 1024 32
echo "MINMin"
./MINMin ${INSTANCES_PATH}/workload.30 ${INSTANCES_PATH}/scenario.19 1024 32
echo "MINMIN"
./MINMIN ${INSTANCES_PATH}/workload.30 ${INSTANCES_PATH}/scenario.19 1024 32

