for instance_dir in $(ls -d */)
do
    echo "Procesando ${instance_dir}"
    cd ${instance_dir}

	/home/siturria/AE/metricas_mo/FP_2obj data_12.log
	mv FP.out FP_12.out

	/home/siturria/AE/metricas_mo/FP_2obj data_13.log
	mv FP.out FP_13.out
	
	/home/siturria/AE/metricas_mo/FP_2obj data_23.log
	mv FP.out FP_23.out

	/home/siturria/AE/metricas_mo/FP_3obj data_00.log
	mv FP.out FP_00.out

    cd ..
done




