for instance_dir in $(ls -d */)
do
    echo "Procesando ${instance_dir}"
    cd ${instance_dir}

    trial=0
    for trial_dir in $(ls -d */)
    do
        echo "> ${trial_dir} (${trial})"
        cat ${trial_dir}/*.sol_* > results_${trial}.txt
    	trial=$(($trial + 1))
    done

    cat */*.sol_* > results.txt
    cd ..
done

