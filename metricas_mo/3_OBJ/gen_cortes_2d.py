'''
Created on Oct 14, 2011

@author: santiago
'''

import sys
import os

if __name__ == '__main__':
	for path in os.listdir("."):
		if os.path.isdir(path):
			print "[PROCESSING] " + path

			inputfile = open(os.path.join(path, "data.log"),"r")

			output0 = open(os.path.join(path, "data_00.log"),"w")
			output1 = open(os.path.join(path, "data_12.log"),"w")
			output2 = open(os.path.join(path, "data_13.log"),"w")
			output3 = open(os.path.join(path, "data_23.log"),"w")

			for line in inputfile:
				data = str(line).strip().split(" ")
				print data
				output0.write(data[0] + " " + data[1] + " " + data[2] + "\n")
				output1.write(data[0] + " " + data[1] + "\n")
				output2.write(data[0] + " " + data[2] + "\n")
				output3.write(data[1] + " " + data[2] + "\n")

			inputfile.close()
			output0.close()
			output1.close()
			output2.close()

