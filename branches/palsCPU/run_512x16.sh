INSTANCES_PATH=/home/santiago/Scheduling/Instances/Makespan-Energy/512x16
#valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --num-callers=20 --track-fds=yes bin/pals_cpu ${INSTANCES_PATH}/scenario.19 ${INSTANCES_PATH}/workload.30 512 16 1 1 0
bin/pals_cpu ${INSTANCES_PATH}/scenario.19 ${INSTANCES_PATH}/workload.30 512 16 1 1 0
