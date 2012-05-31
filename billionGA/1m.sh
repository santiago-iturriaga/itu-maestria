rm bin/billionga
svn update
make

export LD_LIBRARY_PATH=/home/clusterusers/siturriaga/cuda/lib64:/home/clusterusers/siturriaga/cuda/lib
export OMP_NUM_THREADS=2

time (bin/billionga 1048576 1000000 524288 0 > out.1m.txt) &> out.1m.time
time (bin/billionga 1048576 1000000 524288 0 > out.1m.txt) &> out.1m.time

