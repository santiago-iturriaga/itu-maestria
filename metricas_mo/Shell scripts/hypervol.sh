for instance_dir in $(ls -d */)
do
    echo "Procesando ${instance_dir}"
    cd ${instance_dir}

    matlab < hypervolume_run.m > reporte_hypervol.txt
    
    cd ..
done

