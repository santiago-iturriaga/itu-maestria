TASKS=8
MACHINES=2

./limpiar_todo.sh
mkdir ${TASKS}x${MACHINES}

python2 generador_scenario.py ${MACHINES} 4 30784
python2 generador_workload.py ${TASKS} ${MACHINES} 1 1 28672
python2 generador_workload.py ${TASKS} ${MACHINES} 2 1 5190
python2 generador_workload.py ${TASKS} ${MACHINES} 3 1 4269
python2 generador_priorities.py ${TASKS} 1 31079
python2 generador_cores.py ${TASKS} 4 24667
python2 generador_arrivals.py ${TASKS} 0.05 1 19423

mv cores_*.* ${TASKS}x${MACHINES}
mv priorities.* ${TASKS}x${MACHINES}
mv scenario_*.* ${TASKS}x${MACHINES}
mv workload_*.* ${TASKS}x${MACHINES}
mv arrival.* ${TASKS}x${MACHINES}
