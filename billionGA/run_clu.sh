rm bin/billonga
#svn update
make

export LD_LIBRARY_PATH=/home/clusterusers/siturriaga/cuda/lib64:/home/clusterusers/siturriaga/cuda/lib
export OMP_NUM_THREADS=2

#export OMP_NUM_THREADS=1
#time (bin/billionga 1024 50 512 0 1> out.std 2> out.err) &> out.time

#time(bin/billionga 1073741824 10000 268435456 0) > out-1b.txt
#bin/billionga 1048576 2000000 524288 0
#bin/billionga 1048576 2000000 1048576 0

#bin/billionga 1048576 2000000 524288 0
#time (bin/billionga 1048576 2000000 524288 0 > out.txt) &> out.time

time (bin/billionga 8388608 1000000 4194304 0 > out.txt) &> out.time
