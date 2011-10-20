for instance_dir in $(ls -d */)
do
    echo "Procesando ${instance_dir}"
    cd ${instance_dir}

		ln -s /home/siturria/AE/metricas_mo/3_OBJ/gen_cortes_2d.py .
		ln -s /home/siturria/AE/metricas_mo/3_OBJ/gnuplot.dat .
		ln -s /home/siturria/AE/metricas_mo/3_OBJ/mass_cat_obj.sh .
		ln -s /home/siturria/AE/metricas_mo/3_OBJ/mass_fp_obj.sh .
		ln -s /home/siturria/AE/metricas_mo/3_OBJ/mass_plot.obj.sh .

    cd ..
done

