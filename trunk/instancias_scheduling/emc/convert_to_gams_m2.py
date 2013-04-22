#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  convert_to_gams.py
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

scale = 1
#scale = 100

def main():
    if len(sys.argv) != 8:
        print("Usage %s <#tasks> <#machines> <arrival_file> <priorities_file> <workload_file> <cores_file> <scenario_file>".format(sys.argv[0]))
        sys.exit()

    dim_t = int(sys.argv[1]) #8
    dim_m = int(sys.argv[2]) #2

    scenario_file = sys.argv[7] #"8x2/scenario_c4_high.1"
    arrival_file = sys.argv[3] #"8x2/arrival.0"
    cores_file = sys.argv[6] #"8x2/cores_c2.0"
    priorities_file = sys.argv[4] #"8x2/priorities.0"
    workload_file = sys.argv[5] #"8x2/workload_high.0"

    scenario = []
    arrival = []
    cores = []
    priorities = []
    workload = []

    with open(scenario_file) as f:
        for line in f:
            data_array = line.strip().split('\t')
            assert(len(data_array)==4)
            scenario.append((int(data_array[0]),float(data_array[1]),float(data_array[2])/scale,float(data_array[3])/scale))

    #print(scenario)

    with open(arrival_file) as f:
        for line in f:
            data_array = line.strip().split('\t')
            assert(len(data_array)==2)
            for i in range(int(data_array[1])):
                arrival.append(float(data_array[0])/scale)

    #print(arrival)

    with open(cores_file) as f:
        for line in f:
            cores.append(int(line.strip()))

    #print(cores)

    with open(priorities_file) as f:
        for line in f:
            priorities.append(int(line.strip()))

    #print(priorities)

    with open(workload_file) as f:
        for line in f:
            workload.append(float(line.strip())/scale)

    #print(workload)

    max_cores = 0

    for i in scenario:
        if i[0] > max_cores: max_cores = i[0]

    print("SET t /0*" + str(dim_t+1) + "/;")
    print("SET m /1*" + str(dim_m) + "/;")

    print("PARAMETER m_cores(m)")
    index = 1
    for i in scenario:
        if index == 1:
            print("\t/\t" + str(index) + "\t" + str(i[0]))
        elif index == len(scenario):
            print("\t\t" + str(index) + "\t" + str(i[0]) + " /;")
        else:
            print("\t\t" + str(index) + "\t" + str(i[0]))

        index = index + 1

    print("PARAMETER t_arrival(t)")
    index = 1
    for i in arrival:
        if index == 1:
            print("\t/\t" + str(index) + "\t" + str(i))
        elif index == len(cores):
            print("\t\t" + str(index) + "\t" + str(i) + " /;")
        else:
            print("\t\t" + str(index) + "\t" + str(i))
            
        index = index + 1

    print("PARAMETER t_cores(t)")
    index = 1
    for i in cores:
        if index == 1:
            print("\t/\t" + str(index) + "\t" + str(i))
        elif index == len(cores):
            print("\t\t" + str(index) + "\t" + str(i) + " /;")
        else:
            print("\t\t" + str(index) + "\t" + str(i))
            
        index = index + 1
        
    print("PARAMETER t_priorities(t)")
    index = 1
    for i in priorities:
        if index == 1:
            print("\t/\t" + str(index) + "\t" + str(i))
        elif index == len(cores):
            print("\t\t" + str(index) + "\t" + str(i) + " /;")
        else:
            print("\t\t" + str(index) + "\t" + str(i))
            
        index = index + 1
        
    print("PARAMETER etc(t,m)")
    i_idx = 1
    for i in workload:
        j_idx = 1
        for j in scenario:
            if i_idx == 1 and j_idx == 1:
                print("\t/\t" + str(i_idx) + "\t." + str(j_idx) + "\t{0:.4f}".format(i / (j[1] / j[0])))
            elif i_idx == len(workload) and j_idx == len(scenario):
                print("\t\t" + str(i_idx) + "\t." + str(j_idx) + "\t{0:.4f} /;".format(i / (j[1] / j[0])))
            else:
                print("\t\t" + str(i_idx) + "\t." + str(j_idx) + "\t{0:.4f}".format(i / (j[1] / j[0])))
            
            j_idx = j_idx + 1
        i_idx = i_idx + 1

    print("PARAMETER m_eidle(m)")
    index = 1
    for i in scenario:
        if index == 1:
            print("\t/\t" + str(index) + "\t" + str(i[2]))
        elif index == len(scenario):
            print("\t\t" + str(index) + "\t" + str(i[2]) + " /;")
        else:
            print("\t\t" + str(index) + "\t" + str(i[2]))
            
        index = index + 1

    print("PARAMETER eec(t,m)")
    i_idx = 1
    for i in workload:
        j_idx = 1
        for j in scenario:
            etc_ij = i / (j[1] / j[0])
            inc_per_core_j = (j[3] - j[2]) / j[0]
            eec_ij = etc_ij * inc_per_core_j * cores[i_idx-1]
            
            if i_idx == 1 and j_idx == 1:
                print("\t/\t" + str(i_idx) + "\t." + str(j_idx) + "\t{0:.4f}".format(eec_ij))
            elif i_idx == len(workload) and j_idx == len(scenario):
                print("\t\t" + str(i_idx) + "\t." + str(j_idx) + "\t{0:.4f} /;".format(eec_ij))
            else:
                print("\t\t" + str(i_idx) + "\t." + str(j_idx) + "\t{0:.4f}".format(eec_ij))
            
            j_idx = j_idx + 1
        i_idx = i_idx + 1

    return 0

if __name__ == '__main__':
    main()

