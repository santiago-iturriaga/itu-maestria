#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  fp-eval.py
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
import array

def dominancia(sol_a, sol_b):
    assert(len(sol_a)==len(sol_b))
    
    if sol_a[0] <= sol_b[0] and sol_a[1] >= sol_b[1] and sol_a[2] <= sol_b[2]:
        return 1
    elif sol_a[0] >= sol_b[0] and sol_a[1] <= sol_b[1] and sol_a[2] >= sol_b[2]:
        return -1
    else:
        return 0

def main():
    if len(sys.argv) != 4:
        print("[ERROR] Usage: {0} <computed PF> <num. exec.> <min. coverage>".format(sys.argv[0]))
        exit(-1)
    
    comp_pf_file = sys.argv[1]
    num_exec = int(sys.argv[2])
    min_cover = float(sys.argv[3])

    #print("Computed PF file: {0}".format(comp_pf_file))
    #print("Num. executions : {0}".format(num_exec))
    #print("Min. coverage   : {0}".format(min_cover))
    #print()

    comp_pf_final = []
    num_pf = []
    
    for e in range(num_exec):
        curr_num_pf = 0
        
        with open(comp_pf_file + "." + str(e) + ".out") as f:
            for line in f:
                if len(line.strip()) > 0:
                    data = line.strip().split(",")

                    if len(data) == 10:
                        if (data[0] != "id"):
                            energy = float(data[-4])
                            coverage = float(data[-3])
                            nforwardings = float(data[-2])

                            if coverage > min_cover:
                                comp_pf_final.append((energy,coverage,nforwardings))
                                curr_num_pf = curr_num_pf + 1
                                
        num_pf.append(curr_num_pf)

    global_pf = []
    domination=array.array('i',(0,)*1000)
    
    for i in range(len(comp_pf_final)):
        j = i+1
        
        while domination[i]==0 and j<len(comp_pf_final):
            result = dominancia(comp_pf_final[i], comp_pf_final[j])
            
            if result == 0:
                # Ninguno es dominado por el otro
                pass
            elif result == -1:
                # El primero es dominado
                domination[i] = -1;
            elif result == 1:
                # El primero domina
                domination[j] = -1;
                
            j = j+1

        if domination[i]==0:
            global_pf.append(comp_pf_final[i])

    #print("Computed PF [count={0}]".format(len(comp_pf_final)))
    #for i in comp_pf_final: print("{0:.4f} {1:.4f} {2:.4f}".format(i[0],i[1],i[2]))
    #print()
    #print("Computed PF [count={0}]".format(len(global_pf)))
    for i in global_pf: print("{0:.4f} {1:.4f} {2:.4f}".format(i[0],i[1],i[2]))
    #print()
    #print("Number of solutions:")
    #print(num_pf)
    #sum_num_pf=0
    #for s in num_pf: sum_num_pf = sum_num_pf + s
    #print("Average={0}".format(sum_num_pf / len(num_pf)))
    #print()
    
    return 0

if __name__ == '__main__':
    main()

