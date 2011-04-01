for instance_dir in $(ls -d */)
do
    echo "Procesando ${instance_dir}"
    cd ${instance_dir}

    ln -s ../hypervolume.m hypervolume.m
    ln -s ../hypervolume_run.m hypervolume_run.m
    
    cd ..
done

