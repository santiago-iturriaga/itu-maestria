for instance_dir in $(ls -d */)
do
    echo "Procesando ${instance_dir}"
    cd ${instance_dir}

    cat gd_*.txt > reporte_gd.txt

    cd ..
done

