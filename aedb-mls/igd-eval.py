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

def euclidean_distance(point_a, point_b):
    distance = 0.0

    assert(len(point_a) == len(point_b))
    dimension = len(point_a)

    for coord in range(dimension):
        distance = distance + pow(point_a[coord]-point_b[coord],2)

    return math.sqrt(distance)

def closest_point(p1, list_p):
    min_sol = None
    min_sol_dist = None

    for p2 in list_p:
        if min_sol is None:
            min_sol = p2
            min_sol_dist = euclidean_distance(p1, p2)
        else:
            aux_distance = euclidean_distance(p1, p2)

            if aux_distance < min_sol_dist:
                min_sol = p2
                min_sol_dist = aux_distance

    return min_sol_dist

def igd(pf, approx_pf):
    #print("PF:")
    #print(pf)
    #print("Approx PF:")
    #print(approx_pf)
    
    partial_sum = 0
    
    for pf_s in pf:
        d = closest_point(pf_s, approx_pf)
        partial_sum = partial_sum + pow(d,2)
        
    return math.sqrt(partial_sum) / len(pf)

def main():
    if len(sys.argv) != 6:
        print("[ERROR] Usage: {0} <best PF> <moea PF> <computed PF> <num. exec.> <min. coverage>".format(sys.argv[0]))
        exit(-1)

    best_pf_file = sys.argv[1]
    modea_pf_file = sys.argv[2]
    comp_pf_file = sys.argv[3]
    num_exec = int(sys.argv[4])
    min_cover = float(sys.argv[5])

    print("Best PF file    : {0}".format(best_pf_file))
    print("MOEA PF file    : {0}".format(modea_pf_file))
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

    with open(modea_pf_file) as f:
        for line in f:
            if len(line.strip()) > 0:
                data = line.strip().split(" ")
                assert(len(data)==3)

                if float(data[1]) >= min_cover:
                    moea_pf.append((float(data[0]),float(data[1]),float(data[2])))

    moea_igd = igd(best_pf, moea_pf)

    igd_list = []
    for i in range(30):
        igd_list.append([])

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

        for i in range(len(comp_pf)):
            igd_value = igd(best_pf, comp_pf[i])
            igd_list[i].append(igd_value)
            num_sols_pf[i].append(nd_pf[i])

        igd_value = igd(best_pf, comp_pf_final)
        for i in range(len(comp_pf),30):
            igd_list[i].append(igd_value)
            num_sols_pf[i].append(len(comp_pf_final))

    print("   Average igd, Average ND")
    for i in range(30):
        sum_i = 0
        for j in range(len(igd_list[i])): sum_i = sum_i + igd_list[i][j]

        sum_pf = 0
        for j in range(len(num_sols_pf[i])): sum_pf = sum_pf + num_sols_pf[i][j]

        if len(igd_list[i]) > 0 and len(num_sols_pf[i]):
            print("[{0}] {1:.4f} {2:.4f}".format(i,(sum_i/len(igd_list[i]))/moea_igd,sum_pf/len(num_sols_pf[i])))

    return 0

if __name__ == '__main__':
    main()

