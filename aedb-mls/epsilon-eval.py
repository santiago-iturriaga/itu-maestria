#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  epsilon.py
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
import math
import re

obj = (0,1,0) # min, max, min

def euclidean_distance(point_a, point_b):
    distance = 0.0

    assert(len(point_a) == len(point_b))
    dimension = len(point_a)

    for coord in range(dimension):
        distance = distance + pow(point_a[coord]-point_b[coord],2)

    return math.sqrt(distance)

def epsilon_metric(ref_pf, comp_pf):
    assert(len(ref_pf) > 0)
    assert(len(comp_pf) > 0)

    min_distances = []

    for comp_sol in comp_pf:
    
        min_sol = None
        min_sol_dist = None

        for ref_sol in ref_pf:
            if min_sol is None:
                min_sol = ref_sol
                min_sol_dist = euclidean_distance(ref_sol, comp_sol)
            else:
                aux_distance = euclidean_distance(ref_sol, comp_sol)

                if aux_distance < min_sol_dist:
                    min_sol = comp_sol
                    min_sol_dist = aux_distance

        min_distances.append(min_sol_dist)

    return max(min_distances)

def epsilon_jmetal_metric(ref_pf, comp_pf):
    assert(len(ref_pf) > 0)
    assert(len(comp_pf) > 0)

    eps = None

    for comp_sol in comp_pf:
        eps_j = None
        
        for ref_sol in ref_pf:
            assert(len(comp_sol)==len(ref_sol))
            
            for k in range(len(comp_sol)):
                if obj[k]==0:
                    eps_temp = ref_sol[k]-comp_sol[k]
                else:
                    eps_temp = comp_sol[k]-ref_sol[k]
            
                if k==0:
                    eps_k=eps_temp
                elif eps_k < eps_temp:
                    eps_k=eps_temp
                    
            if eps_j is None:
                eps_j = eps_k
            elif eps_j > eps_k:
                eps_j = eps_k
                
        if eps is None:
            eps = eps_j
        elif eps < eps_j:
            eps = eps_j

    return eps

def main():
    if len(sys.argv) != 6:
        print("[ERROR] Usage: {0} <best PF> <moea PF> <computed PF> <num. exec.> <min. coverage>".format(sys.argv[0]))
        exit(-1)

    best_pf_file = sys.argv[1]
    moea_pf_file = sys.argv[2]
    comp_pf_file = sys.argv[3]
    num_exec = int(sys.argv[4])
    min_cover = float(sys.argv[5])

    print("Best PF file    : {0}".format(best_pf_file))
    print("MOEA PF file    : {0}".format(moea_pf_file))
    print("Computed PF file: {0}".format(comp_pf_file))
    print("Num. executions : {0}".format(num_exec))
    print("Min. coverage   : {0}".format(min_cover))
    print()

    best_pf = []
    with open(best_pf_file) as f:
        for line in f:
            if len(line.strip()) > 0:
                data = line.strip().split(" ")
                assert(len(data)==3)

                if float(data[1]) >= min_cover:
                    best_pf.append((float(data[0]),float(data[1]),float(data[2])))

    moea_pf = []

    with open(moea_pf_file) as f:
        for line in f:
            if len(line.strip()) > 0:
                data = line.strip().split(" ")
                assert(len(data)==3)

                if float(data[1]) >= min_cover:
                    moea_pf.append((float(data[0]),float(data[1]),float(data[2])))

    moea_epsilon = epsilon_metric(best_pf, moea_pf)

    epsilons = []
    for i in range(30):
        epsilons.append([])

    num_sols_pf = []
    for i in range(30):
        num_sols_pf.append([])

    for e in range(num_exec):
        comp_pf_final = []

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

        comp_pf = []
        nd_pf = []
        with open(comp_pf_file + "." + str(e) + ".err") as f:
            print(comp_pf_file + "." + str(e) + ".err")
            line = f.readline()

            while line:
                if line.startswith("[POPULATION]"):
                    data = line.strip().split("=")
                    assert(len(data)==2)

                    count = int(data[1])
                    nd_pf.append(count)

                    current_pf = []

                    for i in range(count):
                        line = f.readline()
                        data = line.strip().split(",")

                        energy = float(data[-4])
                        coverage = float(data[-3])
                        nforwardings = float(data[-2])

                        if coverage > min_cover:
                            current_pf.append((energy,coverage,nforwardings))

                    comp_pf.append(current_pf)

                line = f.readline()

        #print()

        for i in range(len(comp_pf)):
            #epsilon_value = epsilon_jmetal_metric(best_pf, comp_pf[i])
            epsilon_value = epsilon_metric(best_pf, comp_pf[i])
            
            #print("[{0}] Epsilon = {1:.2f}".format(i,epsilon_value))
            epsilons[i].append(epsilon_value)
            num_sols_pf[i].append(nd_pf[i])

        #epsilon_value = epsilon_jmetal_metric(best_pf, comp_pf_final)
        epsilon_value = epsilon_metric(best_pf, comp_pf_final)
        
        for i in range(len(comp_pf),30):
            epsilons[i].append(epsilon_value)
            num_sols_pf[i].append(len(comp_pf_final))

    print("   Average epsilon, Average ND")
    for i in range(30):
        sum_i = 0
        for j in range(len(epsilons[i])): sum_i = sum_i + epsilons[i][j]

        sum_pf = 0
        for j in range(len(num_sols_pf[i])): sum_pf = sum_pf + num_sols_pf[i][j]

        if len(epsilons[i]) > 0 and len(num_sols_pf[i]):
            #print("[{0}] {1:.4f} {2:.1f}".format(i,(sum_i/len(epsilons[i])-moea_epsilon),sum_pf/len(num_sols_pf[i])))
            print("{0:.4f} {1:.1f}".format((sum_i/len(epsilons[i])),sum_pf/len(num_sols_pf[i])))

    #print(epsilons)

    return 0

if __name__ == '__main__':
    main()

