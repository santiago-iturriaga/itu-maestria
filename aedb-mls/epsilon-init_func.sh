FINIT=(mls-comp-seed mls-random mls-ref-seed mls-sub-space)

for (( j=0; j<4; j++ ))
do
	rm ${FINIT[j]}.log

	for (( i=0; i<5; i++ ))
	do
		python epsilon.py best_pf/All3Algs100dev.pf ${FINIT[j]}/${FINIT[j]}.${i} 5 >> ${FINIT[j]}.log
	done
done
