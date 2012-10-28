set -x

cd /home/clusterusers/siturriaga/itu-maestria/trunk/aedb-mls
#make clean
#make

ITERATIONS=20000
THREADS=12
INSTANCE=50
SIMULS=5

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/clusterusers/siturriaga/ns3Files
#time(~/bin/mpich2-1.4.1p1/bin/mpiexec -launcher ssh -launcher-exec oarsh -f $OAR_NODEFILE -np 2 -ppn 1 bin/mls)
time(~/bin/mpich2-1.4.1p1/bin/mpiexec -launcher ssh -launcher-exec oarsh -f $OAR_NODEFILE -np 9 -ppn 1 bin/mls $RANDOM $ITERATIONS $THREADS $SIMULS $INSTANCE)
