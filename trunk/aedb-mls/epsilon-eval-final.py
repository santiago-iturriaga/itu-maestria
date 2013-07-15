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

                if float(data[1]) >= min_cover:
                    best_pf.append((float(data[0]),float(data[1]),float(data[2])))

    epsilons = []
    epsilon_sum = 0.0

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

        epsilon_value = epsilon_metric(best_pf, comp_pf_final)
        epsilons.append(epsilon_value)
        epsilon_sum = epsilon_sum + epsilon_value

    epsilon_average = epsilon_sum / len(epsilons)
    
    epsilon_sqsum = 0.0
    for i in epsilons: epsilon_sqsum = epsilon_sqsum + pow(i-epsilon_average,2)
    epsilon_stdev = math.sqrt(epsilon_sqsum/(len(epsilons)-1))

    print("{0:.4f} {1:.4f}".format(epsilon_average,epsilon_stdev))

    return 0

if __name__ == '__main__':
    main()

