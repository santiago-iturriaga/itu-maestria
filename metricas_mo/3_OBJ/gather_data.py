'''
Created on Oct 14, 2011

@author: santiago
'''

import sys
import os

from subprocess import call

if __name__ == '__main__':
	call(["mkdir","results"])

	for path in os.listdir("."):
		if os.path.isdir(path) and path != "results":
			print "[PROCESSING] " + path

			call(["cp", os.path.join(path, "FP_00.out"), "results/" + path + "_FP_00.out"])
			call(["cp", os.path.join(path, "FP_12.out"), "results/" + path + "_FP_12.out"])
			call(["cp", os.path.join(path, "FP_13.out"), "results/" + path + "_FP_13.out"])
			call(["cp", os.path.join(path, "FP_23.out"), "results/" + path + "_FP_23.out"])

			call(["cp", os.path.join(path, "data_12.png"), "results/" + path + "_data_12.png"])
			call(["cp", os.path.join(path, "data_12.ps"), "results/" + path + "_data_12.ps"])
			call(["cp", os.path.join(path, "data_13.png"), "results/" + path + "_data_13.png"])
			call(["cp", os.path.join(path, "data_13.ps"), "results/" + path + "_data_13.ps"])
			call(["cp", os.path.join(path, "data_23.png"), "results/" + path + "_data_23.png"])
			call(["cp", os.path.join(path, "data_23.ps"), "results/" + path + "_data_23.ps"])

