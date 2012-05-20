export LD_LIBRARY_PATH=/home/clusterusers/siturriaga/cuda/lib64:/home/clusterusers/siturriaga/cuda/lib

export OMP_NUM_THREADS=2
time (bin/billionga 536870912 100000 268435456 0 1> out.std 2> out.err) &> out.time

#export OMP_NUM_THREADS=1
#time(bin/billionga 1073741824 10000 268435456 0) > out-1b.txt
#bin/billionga 1048576 2000000 524288 0
#bin/billionga 1048576 2000000 1048576 0
#time (bin/billionga 1048576 10 1048576 0 > out-1m.txt)
#time (bin/billionga 899999744 2000000 67108864 0 0 > out-899m.txt)
