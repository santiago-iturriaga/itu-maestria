rm bin/billionga
svn update
make cuy

export LD_LIBRARY_PATH=/home/clusterusers/siturriaga/cuda/lib64:/home/clusterusers/siturriaga/cuda/lib
export OMP_NUM_THREADS=4

#time (bin/billionga 1024 50 512 0 1> out.txt) &> out.time

#bin/billionga 1048576 2000000 524288 0
#bin/billionga 1048576 2000000 1048576 0
#bin/billionga 1048576 2000000 524288 0

#time (bin/billionga 1048576 500000 524288 0 > out.txt) &> out.time
time (bin/billionga 1048576 1000000 262144 0 > out.1m.txt) &> out.1m.time
#time (bin/billionga 8388608 1000000 4194304 0 > out.txt) &> out.time
#time (bin/billionga 16777216 1000000 8388608 0 > out.txt) &> out.time
#time (bin/billionga 33554432 1000000 16777216 0 > out.txt) &> out.time
