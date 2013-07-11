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

def get_w_point(true_pf):
    max_obj_1 = true_pf[0][0]
    min_obj_2 = true_pf[0][1]
    max_obj_3 = true_pf[0][2]

    for p in true_pf:
        if p[0] > max_obj_1:
            max_obj_1 = p[0]
        if p[1] < min_obj_2:
            min_obj_2 = p[1]
        if p[2] > max_obj_3:
            max_obj_3 = p[2]
            
    #return (max_obj_1 * 1.1, max_obj_2 * 1.1, max_obj_3 * 1.1)
    return (max_obj_1, min_obj_2, max_obj_3)
    #return (max_obj_1, max_obj_2)
    #return (float(5.420153), float(9293.50293))

def compute_hypervolume(true_pf, inst_pf):
    w = get_w_point(true_pf)

    #print "W point:"
    #print w

    true_pf = sorted(true_pf)
    inst_pf = sorted(inst_pf)

    #print true_pf
    #print inst_pf

    true_pf_hv = (w[0] - true_pf[0][0]) * (w[1] - true_pf[0][1])
    for p in range(len(true_pf)-1):
        hvol = (true_pf[p][1] - true_pf[p+1][1]) * (w[0] - true_pf[p+1][0])
        true_pf_hv = true_pf_hv + hvol

    inst_pf_hv = (w[0] - inst_pf[0][0]) * (w[1] - inst_pf[0][1])
    #print "(w[0] - inst_pf[0][0]) * (w[1] - inst_pf[0][1]) = (%.2f - %.2f) * (%.2f - %.2f) = %.2f" % (w[0], inst_pf[0][0], w[1], inst_pf[0][1], inst_pf_hv)
    for p in range(len(inst_pf)-1):
        hvol = (inst_pf[p][1] - inst_pf[p+1][1]) * (w[0] - inst_pf[p+1][0])
        #print "(inst_pf[p][1] - inst_pf[p+1][1]) * (w[0] - inst_pf[p+1][0]) = (%.2f - %.2f) * (%.2f - %.2f) = %.2f" % (inst_pf[p][1], inst_pf[p+1][1], w[0], inst_pf[p+1][0], hvol)
        inst_pf_hv = inst_pf_hv + hvol
        #print "%.2f" % (inst_pf_hv)

    #print "Hypervolume true PF: %s" % (true_pf_hv)
    #print "Hypervolume inst PF: %s (%s%%)" % (inst_pf_hv, inst_pf_hv / true_pf_hv * 100)

    if (true_pf_hv > 0):
        return inst_pf_hv / true_pf_hv
    else:
        return 0

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

def main():
    if len(sys.argv) != 5:
        print("[ERROR] Usage: {0} <best PF> <computed PF> <num. exec.> <min. coverage>".format(sys.argv[0]))
        exit(-1)

    best_pf_file = sys.argv[1]
    comp_pf_file = sys.argv[2]
    num_exec = int(sys.argv[3])
    min_cover = float(sys.argv[4])

    print("Best PF file    : {0}".format(best_pf_file))
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

                if (float(data[1])*-1) >= min_cover:
                    best_pf.append((float(data[0]),float(data[1])*-1,float(data[2])))

    #print("Best PF [count={0}]".format(len(best_pf)))
    #print(best_pf)
    #print()

    epsilons = []
    for i in range(27):
        epsilons.append([])

    num_sols_pf = []
    for i in range(27):
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

        #print("Computed PF [count={0}]".format(len(comp_pf_final)))
        #print(comp_pf_final)
        #print()

        #comp_pf_index = 0
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
                    #print("INDEX={0} COUNT={1}".format(comp_pf_index, count))

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
                    #comp_pf_index = comp_pf_index + 1

                line = f.readline()

        #print()

        for i in range(len(comp_pf)-1):
            epsilon_value = epsilon_metric(best_pf, comp_pf[i])
            #print("[{0}] Epsilon = {1:.2f}".format(i,epsilon_value))
            epsilons[i].append(epsilon_value)
            num_sols_pf[i].append(nd_pf[i])

        epsilon_value = epsilon_metric(best_pf, comp_pf_final)
        #print("Final Epsilon = {0:.2f}".format(epsilon_value))
        epsilons[len(comp_pf)-1].append(epsilon_value)
        num_sols_pf[len(comp_pf)-1].append(len(comp_pf_final))

    print("   Average epsilon, Average ND")
    for i in range(27):
        sum_i = 0
        for j in range(len(epsilons[i])): sum_i = sum_i + epsilons[i][j]

        sum_pf = 0
        for j in range(len(num_sols_pf[i])): sum_pf = sum_pf + num_sols_pf[i][j]

        if len(epsilons[i]) > 0 and len(num_sols_pf[i]):
            print("[{0}] {1:.4f} {2:.4f}".format(i,sum_i/len(epsilons[i]),sum_pf/len(num_sols_pf[i])))

    #print(epsilons)

    return 0

if __name__ == '__main__':
    main()

