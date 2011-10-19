for instance_dir in $(ls -d */)
do
    echo "Procesando ${instance_dir}"
    cd ${instance_dir}

	ls */*.sol_*_metricas | xargs cat > data.log

    cd ..
done




