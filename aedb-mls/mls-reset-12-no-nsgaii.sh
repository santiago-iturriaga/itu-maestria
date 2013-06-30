set -x

TEST_NAME="mls-reset-12-n-nsgaii"

MPI_DIR=/home/clusterusers/siturriaga/bin/mpich2-1.4.1p1
NS3_DIR=/home/clusterusers/siturriaga/ns3Files
HOME_DIR=/home/clusterusers/siturriaga/itu-maestria/trunk/aedb-mls
RESULT_DIR=$HOME_DIR/$TEST_NAME

cd $HOME_DIR
mkdir $RESULT_DIR

make clean
make release

SEED=1 #$RANDOM

ITERATIONS=250
SIMULATIONS=10
DENSITY=25
RESET=12
MIN_COVERAGE=5
ALPHA=0.2
ELITE=1
INIT_FUNCTION=5
REPORT_START=15
REPORT_EVERY=110

POPULATIONS=9
THREADS=12

EXECUTIONS=1

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NS3_DIR

for (( e=0; e<$EXECUTIONS; e++ ))
do
    time($MPI_DIR/bin/mpiexec -launcher ssh -launcher-exec oarsh -f $OAR_NODEFILE \
        -np $POPULATIONS -ppn 1 bin/mls $SEED $ITERATIONS $THREADS $SIMULATIONS $DENSITY $RESET \
        $MIN_COVERAGE $ALPHA $ELITE $INIT_FUNCTION $REPORT_START $REPORT_EVERY \
        1>$RESULT_DIR/$TEST_NAME.$e.out 2>$RESULT_DIR/$TEST_NAME.$e.err) \
        &>$RESULT_DIR/$TEST_NAME.$e.time
done
