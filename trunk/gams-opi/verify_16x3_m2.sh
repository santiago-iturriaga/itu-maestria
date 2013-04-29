INST_PATH=/home/santiago/google-hosting/itu-maestria/svn/trunk/instancias_scheduling/emc/16x3/
#INST_PATH=../emc/16x3/

echo "WQT--------------"

python verify_solution.py 16 3 ${INST_PATH}arrival.0 ${INST_PATH}priorities.0 ${INST_PATH}workload_high.0 \
	${INST_PATH}cores_c4.19 ${INST_PATH}scenario_c6_mid.31 m2_wqt_assign_16x3.txt m2_wqt_stime_16x3.txt

python verify_solution.py 16 3 ${INST_PATH}arrival.1 ${INST_PATH}priorities.1 ${INST_PATH}workload_high.1 \
	${INST_PATH}cores_c4.10 ${INST_PATH}scenario_c6_high.1 m2_wqt_assign_16x3_2.txt m2_wqt_stime_16x3_2.txt

python verify_solution.py 16 3 ${INST_PATH}arrival.2 ${INST_PATH}priorities.2 ${INST_PATH}workload_high.2 \
	${INST_PATH}cores_c4.16 ${INST_PATH}scenario_c4_high.0 m2_wqt_assign_16x3_3.txt m2_wqt_stime_16x3_3.txt

echo "ENERGY-----------"

python verify_solution.py 16 3 ${INST_PATH}arrival.0 ${INST_PATH}priorities.0 ${INST_PATH}workload_high.0 \
	${INST_PATH}cores_c4.19 ${INST_PATH}scenario_c6_mid.31 m2_nrg_assign_16x3.txt m2_nrg_stime_16x3.txt

python verify_solution.py 16 3 ${INST_PATH}arrival.1 ${INST_PATH}priorities.1 ${INST_PATH}workload_high.1 \
	${INST_PATH}cores_c4.10 ${INST_PATH}scenario_c6_high.1 m2_nrg_assign_16x3_2.txt m2_nrg_stime_16x3_2.txt

python verify_solution.py 16 3 ${INST_PATH}arrival.2 ${INST_PATH}priorities.2 ${INST_PATH}workload_high.2 \
	${INST_PATH}cores_c4.16 ${INST_PATH}scenario_c4_high.0 m2_nrg_assign_16x3_3.txt m2_nrg_stime_16x3_3.txt
