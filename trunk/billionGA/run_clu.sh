export LD_LIBRARY_PATH=/home/clusterusers/siturriaga/cuda/lib64:/home/clusterusers/siturriaga/cuda/lib

cd /home/clusterusers/siturriaga/itu-maestria/trunk/billionGA/
rm salida.txt
make clu
time (bin/billionga 1048576 100000 1048576 0 > salida.txt)
