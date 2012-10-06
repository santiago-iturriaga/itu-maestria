set -x

#MinMin
bin/gmochc_cpu ~/instancias/1024x32.ME/scenario.0 ~/instancias/1024x32.ME/workload.A.u_c_hihi 1024 32 1 1 0 1 1 1> minmin_sol_1k.txt 2> minmin_log_1k.txt
#bin/verificador ~/instancias/1024x32.ME/scenario.0 ~/instancias/1024x32.ME/workload.A.u_c_hihi minmin_sol_1k.txt 1024 32 > minmin_1k.txt

#cMOCHC/islands
SEED=0
#THREADS=2
THREADS=8
TIMEOUT=1
#ITERATIONS=20
ITERATIONS=2000
#ITERATIONS=1000
time( bin/gmochc_cpu ~/instancias/1024x32.ME/scenario.0 ~/instancias/1024x32.ME/workload.A.u_c_hihi \
    1024 32 3 ${THREADS} ${SEED} ${TIMEOUT} ${ITERATIONS} 1> sols_1k.txt 2>log_1k.txt)
#bin/verificador ~/Scheduling/Energy-Makespan/instances.ruso/512x16/scenario.0 ~/Scheduling/Energy-Makespan/instances.ruso/512x16/workload.0 sols.txt 512 16
