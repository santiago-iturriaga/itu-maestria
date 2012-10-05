set -x

DIMENSIONS="1024 32"

#THREADS=1
#THREADS=2
THREADS=8

ITERATIONS=1
PALS_ITERATIONS=100000000
PALS_TIMEOUT=15
#PALS_TIMEOUT=90

#PALS_POP_SIZE=6
PALS_POP_SIZE=16
#PALS_POP_SIZE=300

INST="/home/siturria/instancias/1024x32.ME/scenario.0 /home/siturria/instancias/1024x32.ME/workload.A.u_c_hihi"

RAND=1 #$RANDOM
EXEC="bin/pals_cpu ${INST} ${DIMENSIONS} 1 ${THREADS} ${RAND} ${PALS_TIMEOUT} ${PALS_ITERATIONS} ${PALS_POP_SIZE}"
echo ${EXEC}
time (${EXEC} >> me-mls.sols 2> me-mls.info) 2> me-mls.time
#time (${EXEC})

VERIFICADOR="bin/verificador"
EXEC_VERIF="${VERIFICADOR} ${INST} me-mls.sols ${DIMENSIONS}"
echo ${EXEC_VERIF}
${EXEC_VERIF} > me-mls.metrics
