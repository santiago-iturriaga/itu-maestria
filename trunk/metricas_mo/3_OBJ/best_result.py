#encoding: utf-8
'''
Created on Oct 14, 2011

@author: santiago
'''

import sys
import os
import subprocess
import re

if __name__ == '__main__':
	for path in os.listdir("."):
		best_makespan = float(sys.maxint)
		best_wrr = float(sys.maxint)
		best_priority = float(sys.maxint)

		minmin_makespan = float(sys.maxint)
		minmin_wrr = float(sys.maxint)
		minmin_priority = float(sys.maxint)

		if os.path.isdir(path) and path != "results":
			print "[PROCESSING] " + path

			p = subprocess.Popen(["bash", "grep_reference.sh", path], stdout=subprocess.PIPE)
			out, err = p.communicate()

			# print out

			regex = re.compile(">\((.*)\|(.*)\|(.*)\)$", re.MULTILINE)
			found = regex.findall(out)
			
			makespan = float(found[0][0])
			wrr = float(found[0][1])
			energy = float(found[0][2])
			
			if minmin_makespan > makespan:
			        minmin_makespan = makespan
			if minmin_wrr > wrr:
			        minmin_wrr = wrr
			if minmin_priority > energy:
			        minmin_priority = energy
			
			fp_file = open("FP_00","r")
			for line in fp_file:
				data = str(line).strip().split(" ")
				
				makespan = float(data[0])
				wrr = float(data[1])
				energy = float(data[2])
			
			    if best_makespan > makespan:
					best_makespan = makespan
			    if best_wrr > wrr:
					best_wrr = wrr
			    if best_priority > energy:
					best_priority = energy
				
			close(fp_file)
			
			print "<MinMin / Best>\n"
			print "Makespan: " + minmin_makespan + " / " + best_makespan + "\n"
			print "WRR: " + minmin_wrr + " / " + best_wrr + "\n"
			print "Energy: " + minmin_energy + " / " + best_energy + "\n"

				
