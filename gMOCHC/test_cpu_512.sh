set -x

#MCT
#bin/gmochc_cpu ~/Scheduling/Energy-Makespan/instances.ruso/512x16/scenario.0 ~/Scheduling/Energy-Makespan/instances.ruso/512x16/workload.0 512 16 0 1 0 10 10 10 1> sols.txt
#bin/verificador ~/Scheduling/Energy-Makespan/instances.ruso/512x16/scenario.0 ~/Scheduling/Energy-Makespan/instances.ruso/512x16/workload.0 sols.txt 512 16

#MinMin
#bin/gmochc_cpu ~/Scheduling/Energy-Makespan/instances.ruso/512x16/scenario.0 ~/Scheduling/Energy-Makespan/instances.ruso/512x16/workload.0 512 16 1 1 0 10 10 10 1> sols.txt
#bin/verificador ~/Scheduling/Energy-Makespan/instances.ruso/512x16/scenario.0 ~/Scheduling/Energy-Makespan/instances.ruso/512x16/workload.0 sols.txt 512 16

#pMinMin/D
#bin/gmochc_cpu ~/Scheduling/Energy-Makespan/instances.ruso/512x16/scenario.0 ~/Scheduling/Energy-Makespan/instances.ruso/512x16/workload.0 512 16 2 4 0 10 10 10 1> sols.txt
#bin/verificador ~/Scheduling/Energy-Makespan/instances.ruso/512x16/scenario.0 ~/Scheduling/Energy-Makespan/instances.ruso/512x16/workload.0 sols.txt 512 16

#cMOCHC/islands
SEED=0
#THREADS=2
THREADS=12
TIMEOUT=1
#ITERATIONS=20
ITERATIONS=2000
#ITERATIONS=8000
bin/gmochc_cpu ~/Scheduling/Energy-Makespan/instances.ruso/512x16/scenario.0 ~/Scheduling/Energy-Makespan/instances.ruso/512x16/workload.0 \
    512 16 3 ${THREADS} ${SEED} ${TIMEOUT} ${ITERATIONS} 1> sols.txt 2>log.txt
#bin/verificador ~/Scheduling/Energy-Makespan/instances.ruso/512x16/scenario.0 ~/Scheduling/Energy-Makespan/instances.ruso/512x16/workload.0 sols.txt 512 16
