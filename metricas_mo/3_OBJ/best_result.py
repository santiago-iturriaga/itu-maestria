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
		best_energy = float(sys.maxint)

		minmin_makespan = float(sys.maxint)
		minmin_wrr = float(sys.maxint)
		minmin_energy = float(sys.maxint)

		if os.path.isdir(path) and path != "results":
			print "[PROCESSING] " + path

			p = subprocess.Popen(["bash", "grep_reference.sh", path], stdout=subprocess.PIPE)
			out, err = p.communicate()

			# print out

			regex = re.compile(">\((.*)\|(.*)\|(.*)\)$", re.MULTILINE)
			found = regex.findall(out)
			
			makespan = float(found[0][1])
			wrr = float(found[0][0])
			energy = float(found[0][2])
			
			if minmin_makespan > makespan:
			        minmin_makespan = makespan
			if minmin_wrr > wrr:
			        minmin_wrr = wrr
			if minmin_energy > energy:
			        minmin_energy = energy
			
			fp_file = open(os.path.join(path,"FP_00.out"),"r")
			for line in fp_file:
				data = str(line).strip().split(" ")
				
				makespan = float(data[0])
				wrr = float(data[1])
				energy = float(data[2])
			
				if best_makespan > makespan:
					best_makespan = makespan
				if best_wrr > wrr:
					best_wrr = wrr
				if best_energy > energy:
					best_energy = energy
				
			fp_file.close()
			
			print "%s <MinMin / Best>" % path
			print "Makespan: %f / %f (%f)" % (minmin_makespan, best_makespan, 100-(100*best_makespan/minmin_makespan))
			print "WRR: %f / %f (%f)" % (minmin_wrr, best_wrr, 100-(100*best_wrr/minmin_wrr))
			print "Energy: %f / %f (%f)" % (minmin_energy, best_energy, 100-(100*best_energy/minmin_energy))

				
