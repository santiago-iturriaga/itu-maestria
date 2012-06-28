for (( i=0; i<30; i++)) 
do
	echo ${i}
	/home/santiago/itu-maestria/svn/trunk/metricas_mo/Spread_2obj ../../512x16.24_10s/scenario.3.workload.A.u_c_hihi.${i}/pals-aga.scenario.3.workload.A.u_c_hihi.metrics ../../512x16.fp/scenario.3.workload.A.u_c_hihi.fp
done
