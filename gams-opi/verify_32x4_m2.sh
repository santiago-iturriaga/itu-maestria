INST_PATH=/home/santiago/google-hosting/itu-maestria/svn/trunk/instancias_scheduling/emc/32x4/
#INST_PATH=../emc/32x4/

echo "WQT--------------"

python verify_solution.py 32 4 ${INST_PATH}arrival.0 ${INST_PATH}priorities.0 ${INST_PATH}workload_high.0 \
	${INST_PATH}cores_c8.22 ${INST_PATH}scenario_c12_high.2 m2_wqt_assign_32x4.txt m2_wqt_stime_32x4.txt

python verify_solution.py 32 4 ${INST_PATH}arrival.1 ${INST_PATH}priorities.1 ${INST_PATH}workload_high.1 \
	${INST_PATH}cores_c8.21 ${INST_PATH}scenario_c10_mid.40 m2_wqt_assign_32x4_2.txt m2_wqt_stime_32x4_2.txt

python verify_solution.py 32 4 ${INST_PATH}arrival.2 ${INST_PATH}priorities.2 ${INST_PATH}workload_high.2 \
	${INST_PATH}cores_c4.8 ${INST_PATH}scenario_c4_high.1 m2_wqt_assign_32x4_3.txt m2_wqt_stime_32x4_3.txt

echo "ENERGY-----------"

python verify_solution.py 32 4 ${INST_PATH}arrival.0 ${INST_PATH}priorities.0 ${INST_PATH}workload_high.0 \
	${INST_PATH}cores_c8.22 ${INST_PATH}scenario_c12_high.2 m2_nrg_assign_32x4.txt m2_nrg_stime_32x4.txt

python verify_solution.py 32 4 ${INST_PATH}arrival.1 ${INST_PATH}priorities.1 ${INST_PATH}workload_high.1 \
	${INST_PATH}cores_c8.21 ${INST_PATH}scenario_c10_mid.40 m2_nrg_assign_32x4_2.txt m2_nrg_stime_32x4_2.txt

python verify_solution.py 32 4 ${INST_PATH}arrival.2 ${INST_PATH}priorities.2 ${INST_PATH}workload_high.2 \
	${INST_PATH}cores_c4.8 ${INST_PATH}scenario_c4_high.1 m2_nrg_assign_32x4_3.txt m2_nrg_stime_32x4_3.txt
