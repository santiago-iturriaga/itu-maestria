set -x

DIMENSIONS="1024 32"

#THREADS=1
THREADS=2
#THREADS=4

ITERATIONS=1
PALS_ITERATIONS=100000000
PALS_TIMEOUT=30
#PALS_TIMEOUT=90

PALS_POP_SIZE=6
#PALS_POP_SIZE=16
#PALS_POP_SIZE=300

INST="/home/santiago/Scheduling/Energy-Makespan/instances.ruso/1024x32/scenario.0 /home/santiago/Scheduling/Energy-Makespan/instances.ruso/1024x32/workload.0"

RAND=1 #$RANDOM
EXEC="bin/pals_cpu ${INST} ${DIMENSIONS} 1 ${THREADS} ${RAND} ${PALS_TIMEOUT} ${PALS_ITERATIONS} ${PALS_POP_SIZE}"
echo ${EXEC}
time (${EXEC} >> me-mls.sols 2> me-mls.info) 2> me-mls.time
#time (${EXEC})

VERIFICADOR="bin/verificador"
EXEC_VERIF="${VERIFICADOR} ${INST} me-mls.sols ${DIMENSIONS}"
echo ${EXEC_VERIF}
${EXEC_VERIF} > me-mls.metrics
