#32x4/arrival.0 32x4/priorities.0 32x4/workload_high.0 32x4/cores_c8.22 32x4/scenario_c12_high.2

#INST_PATH=/home/santiago/google-hosting/itu-maestria/svn/trunk/instancias_scheduling/emc/16x3/
INST_PATH=../emc/32x4/

python verify_solution.py 32 4 ${INST_PATH}arrival.0 ${INST_PATH}priorities.0 ${INST_PATH}workload_high.0 ${INST_PATH}cores_c8.22 ${INST_PATH}scenario_c12_high.2 m2_assignment_32x4.txt m2_starting_time_32x4.txt

