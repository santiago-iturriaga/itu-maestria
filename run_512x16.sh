INSTANCES_PATH=instancias/512x16
#valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --num-callers=20 --track-fds=yes bin/pals_cpu ${INSTANCES_PATH}/scenario.19 ${INSTANCES_PATH}/workload.30 512 16 1 1 0
#bin/pals_cpu ${INSTANCES_PATH}/scenario.19 ${INSTANCES_PATH}/workload.30 512 16 0 2 0
bin/pals_cpu ${INSTANCES_PATH}/scenario.19 ${INSTANCES_PATH}/workload.30 512 16 1 3 0
#bin/pals_cpu ${INSTANCES_PATH}/scenario.0 ${INSTANCES_PATH}/workload.1 512 16 1 2 0
#bin/pals_cpu ${INSTANCES_PATH}/scenario.19 ${INSTANCES_PATH}/workload.30 512 16 2 2 0
#bin/pals_cpu ${INSTANCES_PATH}/scenario.19 ${INSTANCES_PATH}/workload.30 512 16 3 2 0
