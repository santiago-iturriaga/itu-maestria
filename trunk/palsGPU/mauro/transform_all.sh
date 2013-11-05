#etc_c_8192x256_hihi-1.log
#etc_c_16384x512_hihi-1.log
#etc_c_32768x1024_hihi-1.log

DIMS=3

DIM[0]="8192x256"
DIM[1]="16384x512"
DIM[2]="32768x1024"

INST_NUM=20

mkdir sol

for (( d=0; d<DIMS; d++ ))
do
    for (( i=1; i<=INST_NUM; i++ ))
    do
    
        in_filename="result/etc_c_${DIM[d]}_hihi-${i}.log"
        out_filename="sol/etc_c_${DIM[d]}_hihi-${i}.sol"
    
        echo "${in_filename} -> ${out_filename}"
        (python transform.py ${in_filename}) > ${out_filename}
    done
done
