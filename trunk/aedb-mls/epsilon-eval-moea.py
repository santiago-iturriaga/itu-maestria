#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  epsilon.py
#
#  Copyright 2013  <santiago@marga>
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

if __name__ == '__main__':
    main()


def main():
    if len(sys.argv) != 5:
        print("[ERROR] Usage: {0} <best PF> <moea PF> <num. exec.> <min. coverage>".format(sys.argv[0]))
        exit(-1)

    best_pf_file = sys.argv[1]
    moea_pf_file = sys.argv[2]
    moea_num_exec = int(sys.argv[3])
    min_cover = float(sys.argv[4])

    #print("Best PF file    : {0}".format(best_pf_file))
    #print("MOEA PF file    : {0} ({1})".format(moea_pf_file, moea_num_exec))
    #print("Computed PF file: {0} ({1})".format(comp_pf_file, comp_num_exec))
    #print("Min. coverage   : {0}".format(min_cover))
    #print()

    best_pf = []
    with open(best_pf_file) as f:
        for line in f:
            if len(line.strip()) > 0:
                data = line.strip().split(" ")
                assert(len(data)==3)

                if float(data[1]) >= min_cover:
                    best_pf.append((float(data[0]),float(data[1]),float(data[2])))

    moea_pf = []
    moea_pf_value = []
    for e in range(moea_num_exec):
        moea_pf_exec = []

        with open(moea_pf_file + "." + str(e)) as f:
            for line in f:
                if len(line.strip()) > 0:
                    data = line.strip().split("\t")

                    if len(data) == 3:
                        energy = float(data[0])
                        coverage = -1*float(data[1])
                        nforwardings = float(data[2])

                        if coverage > min_cover and energy > 0:
                            moea_pf_exec.append((energy,coverage,nforwardings))

        moea_pf.append(moea_pf_exec)
        moea_pf_value.append(epsilon_metric(best_pf, moea_pf_exec))

    #print ("===================================")
    #for i in best_pf:
    #    print("{0},{1},{2}".format(i[0],i[1],i[2]))
    #print ("===================================")
    #for i in comp_pf[0]:
    #    print("{0},{1},{2}".format(i[0],i[1],i[2]))
    #print (comp_pf_value[0])
    #print ("===================================")
    #for i in moea_pf[0]:
    #    print("{0},{1},{2}".format(i[0],i[1],i[2]))
    #print (moea_pf_value[0])
    #print ("===================================")

    for i in moea_pf_value:
        print(i)

    return 0

if __name__ == '__main__':
    main()
