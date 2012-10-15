#encoding: utf-8

import sys

if len(sys.argv) != 3:
    exit(-1)
    
directorio = sys.argv[1]
print "Path: %s" % directorio

dimension = sys.argv[2]
print "Dim: %s" % dimension

scenarios = 20
workloads = ("A.u_c_hihi","A.u_c_hilo","A.u_c_lohi","A.u_c_lolo","A.u_i_hihi","A.u_i_hilo","A.u_i_lohi","A.u_i_lolo","A.u_s_hihi","A.u_s_hilo","A.u_s_lohi","A.u_s_lolo","B.u_c_hihi","B.u_c_hilo","B.u_c_lohi","B.u_c_lolo","B.u_i_hihi","B.u_i_hilo","B.u_i_lohi","B.u_i_lolo","B.u_s_hihi","B.u_s_hilo","B.u_s_lohi","B.u_s_lolo")

print "scenario,workload,minmin_makespan,minmin_energy,chc_makespan,chc_energy,improv.makespan,improv.energy"

for s in range(scenarios):
    for w in workloads:
        minmin_file = open(directorio+"/minmin_"+str(s)+"_"+w+"_"+dimension+".metrics",'r')
        minmin_line = minmin_file.read()
        
        minmin_array = minmin_line.strip().split(' ')
        minmin_makespan = float(minmin_array[0])
        minmin_energy = float(minmin_array[1])

        chc_makespan = -1
        chc_energy = -1
        for chc_line in open(directorio+"/chc__"+str(s)+"_"+w+"_"+dimension+".metrics",'r'):
            chc_array = chc_line.strip().split(' ')
            aux_makespan = float(chc_array[0])
            aux_energy = float(chc_array[1])        
            
            if (chc_makespan == -1) or (chc_energy == -1):
                chc_makespan = aux_makespan
                chc_energy = aux_energy
            else:
                if (aux_makespan < chc_makespan):
                    chc_makespan = aux_makespan
                if (aux_energy < chc_energy):
                    chc_energy = aux_energy
        
        print "%d,%s,%f,%f,%f,%f,%.2f,%.2f" % (s, w, minmin_makespan, minmin_energy, chc_makespan, chc_energy, (1-chc_makespan/minmin_makespan)*100, (1-chc_energy/minmin_energy)*100)
