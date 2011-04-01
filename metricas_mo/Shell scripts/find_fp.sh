for instance_dir in $(ls -d */)
do
    echo "Procesando ${instance_dir}"
    cd ${instance_dir}

    trial=0
    for trial_dir in $(ls -d */)
    do
        echo "> ${trial_dir} (${trial})"
        /home/santiago/eclipse/c-c++-workspace/Metricas_MO/FP_2obj_3col results_${trial}.txt > results_${trial}.log
        mv FP.out FP_${trial}.out
        #mv FP.out.disc FP_${trial}.out.disc
	    trial=$(($trial + 1))
    done

    /home/santiago/eclipse/c-c++-workspace/Metricas_MO/FP_2obj_3col results.txt > results.log
    cd ..
done

