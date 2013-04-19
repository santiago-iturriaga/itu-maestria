TASKS=512
MACHINES=16

./limpiar_todo.sh
mkdir ${TASKS}x${MACHINES}

python2 generador_scenario.py ${MACHINES} 60 30784
python2 generador_workload.py ${TASKS} ${MACHINES} 1 10 28672
python2 generador_workload.py ${TASKS} ${MACHINES} 2 10 5190
python2 generador_workload.py ${TASKS} ${MACHINES} 3 10 4269
python2 generador_priorities.py ${TASKS} 10 31079
python2 generador_cores.py ${TASKS} 30 24667
python2 generador_arrivals.py ${TASKS} 0.05 30 19423
#python2 generador_arrivals.py ${TASKS} 0.005 30 19423

mv cores_*.* ${TASKS}x${MACHINES}
mv priorities.* ${TASKS}x${MACHINES}
mv scenario_*.* ${TASKS}x${MACHINES}
mv workload_*.* ${TASKS}x${MACHINES}
mv arrival.* ${TASKS}x${MACHINES}
