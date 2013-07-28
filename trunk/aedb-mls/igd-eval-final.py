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
    if len(sys.argv) != 5:
        print("[ERROR] Usage: {0} <best PF> <computed PF> <num. exec.> <min. coverage>".format(sys.argv[0]))
        exit(-1)

    best_pf_file = sys.argv[1]
    comp_pf_file = sys.argv[2]
    num_exec = int(sys.argv[3])
    min_cover = float(sys.argv[4])

    #print("Best PF file    : {0}".format(best_pf_file))
    #print("Computed PF file: {0}".format(comp_pf_file))
    #print("Num. executions : {0}".format(num_exec))
    #print("Min. coverage   : {0}".format(min_cover))
    #print()

    best_pf = []
    with open(best_pf_file) as f:
        for line in f:
            if len(line.strip()) > 0:
                data = line.strip().split(" ")
                assert(len(data)==3)

                if float(data[1]) >= min_cover:
                    best_pf.append((float(data[0]),1/float(data[1]),float(data[2])))

    igd_sum = 0.0
    igd_list = []

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
                                comp_pf_final.append((energy,1/coverage,nforwardings))

        igd_value = igd(best_pf, comp_pf_final)
        igd_list.append(igd_value)
        igd_sum = igd_sum + igd_value

    for i in igd_list:
        print(i)

    #igd_average = igd_sum / len(igd_list)
    
    #igd_sqsum = 0.0
    #for i in igd_list: igd_sqsum = igd_sqsum + pow(i-igd_average,2)
    #igd_stdev = math.sqrt(igd_sqsum/(len(igd_list)-1))

    #print("{0:.4f} {1:.4f}".format(igd_average,igd_stdev))

    return 0

if __name__ == '__main__':
    main()

