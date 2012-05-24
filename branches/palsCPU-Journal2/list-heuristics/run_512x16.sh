INSTANCES_PATH=/home/santiago/Scheduling/Instances/Makespan-Energy/512x16

#make clean
#make all

#echo "MinMin"
./MinMin ${INSTANCES_PATH}/workload.1 ${INSTANCES_PATH}/scenario.0 512 16
#echo "MinMIN"
./MinMIN ${INSTANCES_PATH}/workload.1 ${INSTANCES_PATH}/scenario.0 512 16
#echo "MINMin"
./MINMin ${INSTANCES_PATH}/workload.1 ${INSTANCES_PATH}/scenario.0 512 16
#echo "MINMIN"
./MINMIN ${INSTANCES_PATH}/workload.1 ${INSTANCES_PATH}/scenario.0 512 16

