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

			regex = re.compile("Min-Min<(.*)\|(.*)\|(.*)>$", re.MULTILINE)
			found = regex.findall(out)

			if minmin_makespan > found[0]:
				minmin_makespan = float(found[0])
			if minmin_wrr > found[1]:
				miminn_wrr = float(found[1])
			if minmin_priority > found[2]:
				minmin_priority = float(found[2])

			fp_file = open("FP_00","r")
			for line in fp_file:
				data = str(line).strip().split(" ")
                output0.write(data[0] + " " + data[1] + " " + data[2] + "\n")
                output1.write(data[0] + " " + data[1] + "\n")
                output2.write(data[0] + " " + data[2] + "\n")
                output3.write(data[1] + " " + data[2] + "\n")
				
