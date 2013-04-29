INST_PATH=/home/santiago/google-hosting/itu-maestria/svn/trunk/instancias_scheduling/emc/8x2/
#INST_PATH=../emc/8x2/

echo "WQT--------------"

python verify_solution.py 8 2 ${INST_PATH}arrival.0 ${INST_PATH}priorities.0 ${INST_PATH}workload_high.0 \
	${INST_PATH}cores_c2.0 ${INST_PATH}scenario_c4_high.1 m2_wqt_assign_8x2.txt m2_wqt_stime_8x2.txt

python verify_solution.py 8 2 ${INST_PATH}arrival.1 ${INST_PATH}priorities.1 ${INST_PATH}workload_high.1 \
	${INST_PATH}cores_c4.15 ${INST_PATH}scenario_c6_high.6 m2_wqt_assign_8x2_2.txt m2_wqt_stime_8x2_2.txt

python verify_solution.py 8 2 ${INST_PATH}arrival.2 ${INST_PATH}priorities.2 ${INST_PATH}workload_high.2 \
	${INST_PATH}cores_c4.21 ${INST_PATH}scenario_c4_high.3 m2_wqt_assign_8x2_3.txt m2_wqt_stime_8x2_3.txt

echo "ENERGY-----------"

python verify_solution.py 8 2 ${INST_PATH}arrival.0 ${INST_PATH}priorities.0 ${INST_PATH}workload_high.0 \
	${INST_PATH}cores_c2.0 ${INST_PATH}scenario_c4_high.1 m2_nrg_assign_8x2.txt m2_nrg_stime_8x2.txt

python verify_solution.py 8 2 ${INST_PATH}arrival.1 ${INST_PATH}priorities.1 ${INST_PATH}workload_high.1 \
	${INST_PATH}cores_c4.15 ${INST_PATH}scenario_c6_high.6 m2_nrg_assign_8x2_2.txt m2_nrg_stime_8x2_2.txt

python verify_solution.py 8 2 ${INST_PATH}arrival.2 ${INST_PATH}priorities.2 ${INST_PATH}workload_high.2 \
	${INST_PATH}cores_c4.21 ${INST_PATH}scenario_c4_high.3 m2_nrg_assign_8x2_3.txt m2_nrg_stime_8x2_3.txt
