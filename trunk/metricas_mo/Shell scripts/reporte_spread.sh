for instance_dir in $(ls -d */)
do
    echo "Procesando ${instance_dir}"
    cd ${instance_dir}

    cat spread_*.txt > reporte_spread.txt

    cd ..
done

