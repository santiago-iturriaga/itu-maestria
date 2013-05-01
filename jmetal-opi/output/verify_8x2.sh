INST_PATH=/home/santiago/google-hosting/itu-maestria/svn/trunk/instancias_scheduling/emc/8x2/

echo "=== 1 =================================="
for (( i = 0; i < 6; i++ ))
do
	python verify_solution.py 8 2 ${INST_PATH}arrival.0 ${INST_PATH}priorities.0 ${INST_PATH}workload_high.0 \
		${INST_PATH}cores_c2.0 ${INST_PATH}scenario_c4_high.1 OPIStudy/data/NSGAII_OPI/FE-HCSP_8x2_1/VAR.${i}
done

echo "=== 2 =================================="
for (( i = 0; i < 6; i++ ))
do
	python verify_solution.py 8 2 ${INST_PATH}arrival.1 ${INST_PATH}priorities.1 ${INST_PATH}workload_high.1 \
		${INST_PATH}cores_c4.15 ${INST_PATH}scenario_c6_high.6 OPIStudy/data/NSGAII_OPI/FE-HCSP_8x2_2/VAR.${i}
done

echo "=== 3 =================================="
for (( i = 0; i < 6; i++ ))
do
	python verify_solution.py 8 2 ${INST_PATH}arrival.2 ${INST_PATH}priorities.2 ${INST_PATH}workload_high.2 \
		${INST_PATH}cores_c4.21 ${INST_PATH}scenario_c4_high.3 OPIStudy/data/NSGAII_OPI/FE-HCSP_8x2_3/VAR.${i}
done
