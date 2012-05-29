rm bin/billionga
svn update
make cuy

export LD_LIBRARY_PATH=/home/clusterusers/siturriaga/cuda/lib64:/home/clusterusers/siturriaga/cuda/lib
export OMP_NUM_THREADS=4

#1000 = 5m
#5000 = 30m
#10000 = 60m
#100000 = 580m
 
#time (bin/billionga 1073741824 1000 268435456 0 > out.1b.txt) &> out.1b.time
time (bin/billionga 1073741824 100000 268435456 0 > out.1b.txt) &> out.1b.time
