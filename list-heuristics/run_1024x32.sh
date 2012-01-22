INSTANCES_PATH=/home/santiago/Scheduling/Instances/Makespan-Energy/1024x32

make clean
make all

echo "MinMin"
./MinMin ${INSTANCES_PATH}/workload.1 ${INSTANCES_PATH}/scenario.0 1024 32
echo "MinMIN"
./MinMIN ${INSTANCES_PATH}/workload.1 ${INSTANCES_PATH}/scenario.0 1024 32
echo "MINMin"
./MINMin ${INSTANCES_PATH}/workload.1 ${INSTANCES_PATH}/scenario.0 1024 32
echo "MINMIN"
./MINMIN ${INSTANCES_PATH}/workload.1 ${INSTANCES_PATH}/scenario.0 1024 32

