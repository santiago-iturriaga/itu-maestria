#encoding: utf-8

import sys

if sys.argc != 3:
    exit(-1)
    
directorio = sys.argv[1]
print "Path: %s" % directorio

dimension = sys.argv[2]
print "Dim: %s" % dimension

scenarios = 20
workloads = ("A.u_c_hihi","A.u_c_hilo","A.u_c_lohi","A.u_c_lolo","A.u_i_hihi","A.u_i_hilo","A.u_i_lohi","A.u_i_lolo","A.u_s_hihi","A.u_s_hilo","A.u_s_lohi","A.u_s_lolo","B.u_c_hihi","B.u_c_hilo","B.u_c_lohi","B.u_c_lolo","B.u_i_hihi","B.u_i_hilo","B.u_i_lohi","B.u_i_lolo","B.u_s_hihi","B.u_s_hilo","B.u_s_lohi","B.u_s_lolo")

for s in range(scenarios):
    for w in workloads:
        minmin_file = open(directorio+"/minmin_"+str(s)+"_"+w+"_"+dimension+".metrics",'r')
        minmin_line = minmin_file.read()
        
        minmin_array = minmin_line.strip().split(' ')
        minmin_makespan = float(minmin_array[0])
        minmin_energy = float(minmin_array[1])
        
        #chc__6_A.u_s_hihi_1k.metrics
