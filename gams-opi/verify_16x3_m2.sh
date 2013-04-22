#INST_PATH=/home/santiago/google-hosting/itu-maestria/svn/trunk/instancias_scheduling/emc/16x3/
INST_PATH=../emc/16x3/

python verify_solution.py 16 3 ${INST_PATH}arrival.0 ${INST_PATH}priorities.0 ${INST_PATH}workload_high.0 ${INST_PATH}cores_c4.19 ${INST_PATH}scenario_c6_mid.31 m2_assignment.txt m2_starting_time.txt

