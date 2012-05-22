svn update
make

export LD_LIBRARY_PATH=/home/clusterusers/siturriaga/cuda/lib64:/home/clusterusers/siturriaga/cuda/lib

#export OMP_NUM_THREADS=2
#time (bin/billionga 1048576 3000 524288 0)

#export OMP_NUM_THREADS=2
#time (bin/billionga 1048576 2000000 524288 0 1> out.std 2> out.err) &> out.time
#time (bin/billionga 1048576 5 524288 0 1> out.std 2> out.err) &> out.time

#export OMP_NUM_THREADS=2
#time (bin/billionga 536870912 100000 268435456 0 1> out.std 2> out.err) &> out.time

#export OMP_NUM_THREADS=2
#time (bin/billionga 33554432 1500000 16777216 0 1> out.std 2> out.err) &> out.time

#export OMP_NUM_THREADS=2
#time (bin/billionga 8388608 5 4194304 0 1> out.std 2> out.err) &> out.time

export OMP_NUM_THREADS=1
time (bin/billionga 8388608 5 8388608 0 1> out.std 2> out.err) &> out.time

#export OMP_NUM_THREADS=1
#time (bin/billionga 1024 50 512 0 1> out.std 2> out.err) &> out.time

