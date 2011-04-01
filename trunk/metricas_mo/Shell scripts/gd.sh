for instance_dir in $(ls -d */)
do
    echo "Procesando ${instance_dir}"
    cd ${instance_dir}

    trial=0
    for trial_dir in $(ls -d */)
    do
        echo "> ${trial_dir} (${trial})"        
        /home/santiago/eclipse/c-c++-workspace/Metricas_MO/GD_2obj FP_${trial}.out FP.out > gd_${trial}.txt
	trial=$(($trial + 1))
    done

    cd ..
done

