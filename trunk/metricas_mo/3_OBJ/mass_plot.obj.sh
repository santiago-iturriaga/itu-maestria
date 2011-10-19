for instance_dir in $(ls -d */)
do
    echo "Procesando ${instance_dir}"
    cd ${instance_dir}

	gnuplot ../gnuplot.dat

    cd ..
done




