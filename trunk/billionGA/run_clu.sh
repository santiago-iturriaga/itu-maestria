export LD_LIBRARY_PATH=/home/clusterusers/siturriaga/cuda/lib64:/home/clusterusers/siturriaga/cuda/lib
export OMP_NUM_THREADS=2

cd /home/clusterusers/siturriaga/itu-maestria/trunk/billionGA/

#rm out-*.txt
#make clu

bin/billionga 1073741824 100000 536870912 0
#bin/billionga 1048576 2000000 524288 0
#bin/billionga 1048576 2000000 1048576 0
#time (bin/billionga 1048576 2000000 1048576 0 0 > out-1m.txt)
#time (bin/billionga 899999744 2000000 67108864 0 0 > out-899m.txt)
