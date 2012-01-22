INSTANCES_PATH=/home/santiago/Scheduling/Instances/Makespan-Energy/512x16

#make clean
#make pals-gdb

bin/pals_cpu ${INSTANCES_PATH}/scenario.0 ${INSTANCES_PATH}/workload.1 512 16 3 3 0

