export OMP_NUM_THREADS=1

#cd /home/siturria/maestria/branches/billionGA-fast/

#rm out-*.txt
#make clu

#time(bin/billionga 1073741824 10000 268435456 0) > out-1b.txt
#bin/billionga 1048576 2000000 524288 0
#bin/billionga 1048576 2000000 1048576 0

#time (bin/billionga 128 10 128 1 > out-1m.txt)
time (bin/billionga 1048576 10000 1048576 1 > out-1m.txt)

#time (bin/billionga 899999744 2000000 67108864 0 0 > out-899m.txt)
