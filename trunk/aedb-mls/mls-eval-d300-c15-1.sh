set -x

TEST_NAME="mls-eval-d300-c15-1"

MPI_DIR=/home/clusterusers/siturriaga/bin/mpich2-1.4.1p1
NS3_DIR=/home/clusterusers/siturriaga/ns3Files
HOME_DIR=/home/clusterusers/siturriaga/itu-maestria/trunk/aedb-mls
RESULT_DIR=$HOME_DIR/$TEST_NAME

cd $HOME_DIR
mkdir $RESULT_DIR

#make clean
#make release

SEED=1 #$RANDOM

ITERATIONS=250
SIMULATIONS=10
DENSITY=75
RESET=50
MIN_COVERAGE=15
ALPHA=0.2
ELITE=1
REPORT_START=15
REPORT_EVERY=110

## Init functions:
## 0) MLS__REF_SEED
## 1) MLS__COMPROMISE_SEED
## 2) MLS__SUBSPACE_BASED
## 3) MLS__RANDOM_BASED
## 4) MLS__COMPROMISE_SEED (no nsga-ii)
## 5) MLS__SUBSPACE_BASED (no nsga-ii)
INIT_FUNCTION=2

POPULATIONS=9
THREADS=12

EXECUTIONS=10

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NS3_DIR

for (( e=2; e<$EXECUTIONS; e++ ))
do
    time($MPI_DIR/bin/mpiexec -launcher ssh -launcher-exec oarsh -f $OAR_NODEFILE \
        -np $POPULATIONS -ppn 1 bin/mls $SEED $ITERATIONS $THREADS $SIMULATIONS $DENSITY $RESET \
        $MIN_COVERAGE $ALPHA $ELITE $INIT_FUNCTION $REPORT_START $REPORT_EVERY \
        1>$RESULT_DIR/$TEST_NAME.$e.out 2>$RESULT_DIR/$TEST_NAME.$e.err) \
        &>$RESULT_DIR/$TEST_NAME.$e.time
done
