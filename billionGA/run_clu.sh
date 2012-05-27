rm bin/billonga
svn update
make

export LD_LIBRARY_PATH=/home/clusterusers/siturriaga/cuda/lib64:/home/clusterusers/siturriaga/cuda/lib
export OMP_NUM_THREADS=2

#time (bin/billionga 131072 500000 131072 0 1> out.txt) &> out.time

#bin/billionga 1048576 2000000 524288 0
#bin/billionga 1048576 2000000 1048576 0
#bin/billionga 1048576 2000000 524288 0

#time (bin/billionga 1048576 500000 524288 0 > out.txt) &> out.time
time (bin/billionga 1048576 2000000 524288 0 > out.1m.txt) &> out.1m.time
#time (bin/billionga 8388608 1000000 4194304 0 > out.txt) &> out.time
#time (bin/billionga 16777216 1000000 8388608 0 > out.txt) &> out.time
time (bin/billionga 33554432 2000000 16777216 0 > out.32m.txt) &> out.32m.time
time (bin/billionga 67108864 2000000 33554432 0 > out.64m.txt) &> out.64m.time
time (bin/billionga 134217728 2000000 67108864 0 > out.128m.txt) &> out.128m.time
time (bin/billionga 1073741824 2000000 536870912 0 > out.1b.txt) &> out.1b.time

