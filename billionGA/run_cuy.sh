#export OMP_NUM_THREADS=1
#time (bin/billionga 1048576 10000 1048576 1 1> out.std 2> out.err)

#export OMP_NUM_THREADS=4
#time (bin/billionga 1048576 100000 262144 0 1> out.std 2> out.err)

export OMP_NUM_THREADS=4
#bin/billionga 536870912 10000 134217728 0
time (bin/billionga 536870912 100000 134217728 0 1> out.std 2> out.err) &> out.time

#export OMP_NUM_THREADS=4
#time (bin/billionga 1073741824 100000 268435456 0 1> out.std 2> out.err)

#time(bin/billionga 1073741824 10000 268435456 0) > out-1b.txt
#bin/billionga 1048576 2000000 524288 0
#bin/billionga 1048576 2000000 1048576 0
#time (bin/billionga 128 10 128 1 > out-1m.txt)
#time (bin/billionga 1048576 10000 1048576 1 > out-1m.txt)
#time (bin/billionga 899999744 2000000 67108864 0 0 > out-899m.txt)
