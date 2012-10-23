set -x

cd /home/clusterusers/siturriaga/itu-maestria/trunk/aedb-mls
make clean
make

ITERATIONS=10000
THREADS=12

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/clusterusers/siturriaga/ns3Files
time(~/bin/mpich2-1.4.1p1/bin/mpiexec -np 2 bin/mls $RANDOM $ITERATIONS $THREADS)
