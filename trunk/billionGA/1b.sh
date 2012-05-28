rm bin/billonga
svn update
make cuy

export LD_LIBRARY_PATH=/home/clusterusers/siturriaga/cuda/lib64:/home/clusterusers/siturriaga/cuda/lib
export OMP_NUM_THREADS=4

#1000 = 5m
#10000 = 

time (bin/billionga 1073741824 5000 268435456 0 > out.1b.txt) &> out.1b.time
#time (bin/billionga 1073741824 1000000 268435456 0 > out.1b.txt) &> out.1b.time
