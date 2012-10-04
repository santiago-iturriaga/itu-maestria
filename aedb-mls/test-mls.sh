cd /home/clusterusers/siturriaga/itu-maestria/trunk/aedb-mls
make mls-gdb

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/clusterusers/siturriaga/ns3Files
~/bin/mpich2-1.4.1p1/bin/mpiexec -np 2 bin/mls 
