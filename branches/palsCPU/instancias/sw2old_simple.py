#encoding: utf-8

import sys
import math

if __name__ == '__main__':
	if len(sys.argv) != 5:
		print "Uso: %s <scenario file> <workload file> <#tasks> <#machines>" % sys.argv[0]
		print "Error!!!"
		exit();
		
	scenario_path = sys.argv[1]
	workload_path = sys.argv[2]
	task_count = sys.argv[3]
	machine_count = sys.argv[4]

	machine_ssj = []

	scenario_file = open(scenario_path)
	for line in scenario_file:
		if len(line.strip()) > 0:
			machine_info = line.strip().split()
			if len(machine_info) == 4:
				machine_ssj.append(float(machine_info[1]))
	
	current_machine = 0
	
	workload_file = open(workload_path)
	for line in workload_file:
		if len(line.strip()) > 0:
			value = float(line)
			print "%s" % (str(value / machine_ssj[current_machine]))
			
			current_machine = int(current_machine + 1) % int(machine_count)
