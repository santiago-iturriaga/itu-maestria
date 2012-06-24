CLASSPATH="/home/siturria/AE/MOScheduling/bin"
NUM_OBJ=2

for instance_dir in $(ls -d */)
do
    echo "Procesando ${instance_dir}"
    cd ${instance_dir}

    trial=0
    for trial_dir in $(ls -d */)
    do
        echo "> ${trial_dir} (${trial})"
	java -classpath $CLASSPATH jmetal.qualityIndicator.InvertedGenerationalDistanceNonNormalized FP_${trial}.out FP_total.out $NUM_OBJ > igd_${trial}.txt
	trial=$(($trial + 1))
    done

    cd ..
done

