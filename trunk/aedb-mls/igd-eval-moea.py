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
#from scipy.stats import mstats

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
    partial_sum = 0
    
    for pf_s in pf:
        d = closest_point(pf_s, approx_pf)
        partial_sum = partial_sum + pow(d,2)
        
    return math.sqrt(partial_sum) / len(pf)

def avg_stdev(values):
    sum_v = 0.0
    
    for v in values:
        sum_v = sum_v + v
        
    avg = sum_v / len(values)
    
    diff_sq = 0.0
    
    for v in values:
        diff_sq = diff_sq + pow(v - avg, 2)
        
    stdev = math.sqrt(diff_sq/(len(values)-1))
    
    return (avg, stdev)

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
        moea_pf_value.append(igd(best_pf, moea_pf_exec))

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

