INST_PATH=/home/santiago/google-hosting/itu-maestria/svn/trunk/instancias_scheduling/emc/8x2/
#INST_PATH=../emc/8x2/

ECHO "WQT--------------------"
python ../verify_solution.py 8 2 ${INST_PATH}arrival.0 ${INST_PATH}priorities.0 ${INST_PATH}workload_high.0 ${INST_PATH}cores_c2.0 ${INST_PATH}scenario_c4_high.1 m2_wqt_assign_8x2.txt m2_wqt_stime_8x2.txt

ECHO "ENERGY-----------------"
python ../verify_solution.py 8 2 ${INST_PATH}arrival.0 ${INST_PATH}priorities.0 ${INST_PATH}workload_high.0 ${INST_PATH}cores_c2.0 ${INST_PATH}scenario_c4_high.1 m2_nrg_assign_8x2.txt m2_nrg_stime_8x2.txt