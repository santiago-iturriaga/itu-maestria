#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  verify_solution.py
#  
#  Copyright 2013 Unknown <santiago@marga>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import sys

DEBUG = False

scale = 1
tolerance = 0.0001

def main():
    if len(sys.argv) != 9:
        print("Usage {0} <#tasks> <#machines> <arrival_file> <priorities_file> <workload_file> <cores_file> <scenario_file> <VAR>".format(sys.argv[0]))
        sys.exit()
    
    #path = "/home/santiago/google-hosting/itu-maestria/svn/trunk/instancias_scheduling/emc/"
    
    dim_t = int(sys.argv[1]) #16
    dim_m = int(sys.argv[2]) #3

    scenario_file = sys.argv[7]                     #path + "16x3/scenario_c6_mid.31"
    arrival_file = sys.argv[3]                      #path + "16x3/arrival.0"
    cores_file = sys.argv[6]                        #path + "16x3/cores_c4.19"
    priorities_file = sys.argv[4]                   #path + "16x3/priorities.0"
    workload_file = sys.argv[5]                     #path + "16x3/workload_high.0"
       
    scenario = []
    arrival = []
    cores = []
    priorities = []
    workload = []
    
    FUN_file = sys.argv[8]   #"FUN"
    
    with open(scenario_file) as f:
        for line in f:
            data_array = line.strip().split('\t')
            assert(len(data_array)==4)
            scenario.append((int(data_array[0]),float(data_array[1]),float(data_array[2])/scale,float(data_array[3])/scale))

    with open(arrival_file) as f:
        for line in f:
            data_array = line.strip().split('\t')
            assert(len(data_array)==2)
            for i in range(int(data_array[1])):
                arrival.append(float(data_array[0])/scale)

    with open(cores_file) as f:
        for line in f:
            cores.append(int(line.strip()))
    
    with open(priorities_file) as f:
        for line in f:
            priorities.append(int(line.strip()))
    
    with open(workload_file) as f:
        for line in f:
            workload.append(float(line.strip())/scale)


    with open(FUN_file) as f:
        for line in f:
            assign = {}
            start = []

            data_array = line.split(" ")
            
            for m_data in data_array:
                m_data_aux = m_data.strip().strip("<").strip(">")
                
                if len(m_data_aux) > 0:
                    m_data_aux_array = m_data_aux.split("|")
                    m_id = int(m_data_aux_array[0])
                    
                    for t_data_index in range(1,len(m_data_aux_array)):
                        t_data_aux = m_data_aux_array[t_data_index]
                        
                        if len(t_data_aux) > 0:
                            t_data_array = t_data_aux.split("#")
                            
                            t_id = int(t_data_array[0])
                            t_st = float(t_data_array[1])
                            
                            assign[t_id] = m_id
                            start.append((t_st,t_id))

            if DEBUG: print(assign)
            if DEBUG: print(start)
            
            machine_ct = []
            machine_et = []
            schedule_wqt = 0.0
            
            for i in range(dim_m):
                machine_ct.append([])
                machine_et.append(0.0)
                
                for j in range(scenario[i][0]):
                    machine_ct[i].append(0.0)
               
            if DEBUG: print(machine_ct)
            if DEBUG: print("================================")
            
            makespan = 0.0
                
            for i in range(dim_t):
                if DEBUG: print(machine_ct)
                
                i_start = start[i][0]
                i_t_id = start[i][1]
                i_m_id = assign[i_t_id]
            
                if DEBUG: print("t_id:{0:d} m_id:{1:d} st:{2:.2f}".format(i_t_id,i_m_id,i_start))
                
                t_etc = workload[i_t_id] / (scenario[i_m_id][1] / scenario[i_m_id][0])
                t_arrival = arrival[i_t_id]
                t_priority = priorities[i_t_id]
                t_cores = cores[i_t_id]
                
                m_cores = scenario[i_m_id][0]
                
                if DEBUG: print("   arrival:{0:.2f}".format(t_arrival))
                
                assert(t_cores <= m_cores)
                assert(i_start >= t_arrival)
                
                if DEBUG: print("    t_cores: {0}".format(t_cores))
                
                for c in range(t_cores):
                    if DEBUG: print("    core[{0}].ct: {1:.2f}".format(c,machine_ct[i_m_id][c]))
                    
                    assert(machine_ct[i_m_id][c] - i_start <= tolerance)
                    machine_ct[i_m_id][c] = i_start + t_etc           
                    machine_et[i_m_id] = machine_et[i_m_id] + t_etc
                    
                machine_ct[i_m_id] = sorted(machine_ct[i_m_id])
                if makespan < machine_ct[i_m_id][-1]: makespan = machine_ct[i_m_id][-1]
                
                schedule_wqt = schedule_wqt + (t_priority * (i_start + t_etc - t_arrival))
                if DEBUG: print("================================")
               
            energy = 0.0
            
            index = 0
            for i in range(dim_m):
                consumption_per_core = (scenario[i][3]-scenario[i][2])/scenario[i][0]
                energy = energy + (machine_et[i] * consumption_per_core) + ((makespan * scenario[i][0] - machine_et[i]) * scenario[i][2])
                
                index = index + 1
                
                if DEBUG: print("[{0}] Execution time: {1:.2f}".format(i, machine_et[i]))
            
            if DEBUG: print("Makespan: {0:.2f}".format(makespan))
            
            if DEBUG: 
                print("WQT   : {0:.4f}".format(schedule_wqt))  
                print("ENERGY: {0:.4f}".format(energy))
            else:
                print("{0:.4f} {1:.4f}".format(schedule_wqt,energy))
        
    return 0

if __name__ == '__main__':
    main()

