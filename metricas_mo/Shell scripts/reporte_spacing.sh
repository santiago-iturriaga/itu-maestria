for instance_dir in $(ls -d */)
do
    echo "Procesando ${instance_dir}"
    cd ${instance_dir}

    cat spacing_*.txt > reporte_spacing.txt

    cd ..
done

