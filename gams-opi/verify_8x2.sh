#INST_PATH=/home/santiago/google-hosting/itu-maestria/svn/trunk/instancias_scheduling/emc/16x3/
INST_PATH=../emc/8x2/

python verify_solution.py 8 2 ${INST_PATH}arrival.0 ${INST_PATH}priorities.0 ${INST_PATH}workload_high.0 ${INST_PATH}cores_c2.0 ${INST_PATH}scenario_c4_high.1 m3_assignment.txt m3_starting_time.txt
