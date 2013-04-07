TASKS=12
MACHINES=3

./limpiar_todo.sh
mkdir ${TASKS}x${MACHINES}

python generador_scenario.py ${MACHINES} 4 30784
python generador_workload.py ${TASKS} ${MACHINES} 1 1 28672
python generador_workload.py ${TASKS} ${MACHINES} 2 1 5190
python generador_workload.py ${TASKS} ${MACHINES} 3 1 4269
python generador_priorities.py ${TASKS} 1 31079
python generador_cores.py ${TASKS} 4 24667
python generador_arrivals.py ${TASKS} 0.005 1 19423

mv cores_*.* ${TASKS}x${MACHINES}
mv priorities.* ${TASKS}x${MACHINES}
mv scenario_*.* ${TASKS}x${MACHINES}
mv workload_*.* ${TASKS}x${MACHINES}
mv arrival.* ${TASKS}x${MACHINES}
