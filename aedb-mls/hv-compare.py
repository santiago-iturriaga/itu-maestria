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
from scipy.stats import mstats

def getMaxValues(true_pf):
    l = len(true_pf[0])
    maxValues = [-float('inf'),]*l
    
    for s in true_pf:
        for d in range(l):
            if maxValues[d] < s[d]:
                maxValues[d] = s[d]
                
    return maxValues
    
def getMinValues(true_pf):
    l = len(true_pf[0])
    minValues = [float('inf'),]*l
    
    for s in true_pf:
        for d in range(l):
            if minValues[d] > s[d]:
                minValues[d] = s[d]
                
    return minValues

def getNormalizedFront(comp_pf, maxValues, minValues):
    norm_pf = []
    
    for s in comp_pf:
        norm_s = []
        
        for d in range(len(s)):
            norm_s.append((s[d]-minValues[d])/(maxValues[d]-minValues[d]))
        
        norm_pf.append(norm_s)
        
    return norm_pf

def invertedFront(normalized_pf):
    inverted_pf = []
    
    for s in normalized_pf:
        inverted_s = []
        
        for d in range(len(s)):
            if s[d] <= 1.0 and s[d] >= 0.0:
                inverted_s.append(1.0-s[d])
            elif s[d] > 1.0:
                inverted_s.append(0.0);
            elif s[d] < 0.0:
                inverted_s.append(1.0);
                
        inverted_pf.append(inverted_s)
    
    return inverted_pf

def dominates(sol_a,sol_b,noObjectives):   
    betterInAllObj = True

    for d in range(noObjectives):
        betterInAllObj = betterInAllObj and (sol_a[d] >= sol_b[d])
    
    return betterInAllObj

def filterNondominatedSet(pf, length_pf, noObjectives):
    n = length_pf
    i = 0
    
    while i < n:
        j = i + 1
        
        end_loop = False
        while j < n and not end_loop:
            if dominates(pf[i],pf[j],noObjectives):
                n = n-1
                aux = pf[j]
                pf[j] = pf[n]
                pf[n] = aux
            elif dominates(pf[j],pf[i],noObjectives):
                n = n-1
                
                aux = pf[i]
                pf[i] = pf[n]
                pf[n] = aux
                
                i = i-1
                end_loop = True
            else:
                j = j+1
                
        i = i+1
        
    return n

def surfaceUnchangedTo(pf, length_pf, obj):
    assert(length_pf >= 1)
    
    minValue = pf[0][obj]
    for i in range(length_pf):
        if pf[i][obj] < minValue:
            minValue = pf[i][obj]

    return minValue

def reduceNondominatedSet(pf, length_pf, obj, threshold):
    n = length_pf
    i = 0
    
    while i < n:
        if pf[i][obj] <= threshold:
            n = n-1
            aux = pf[i]
            pf[i] = pf[n]
            pf[n] = aux
            
        i = i + 1
        
    return n

def hypervolume(pf, length_pf, noObjectives):
    volume = 0
    distance = 0
    
    n = length_pf
    
    while n > 0:
        numberND = filterNondominatedSet(pf, n, noObjectives-1)

        tempVolume=0
        if (noObjectives < 3):
            assert(numberND >= 1)
            tempVolume = pf[0][0]
        else:
            tempVolume = hypervolume(pf, numberND, noObjectives-1)
            
        tempDistance = surfaceUnchangedTo(pf, n, noObjectives-1)
        volume = volume + (tempVolume*(tempDistance-distance))
        distance = tempDistance
        n = reduceNondominatedSet(pf, n, noObjectives-1, distance)
            
    return volume

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
    if len(sys.argv) != 7:
        print("[ERROR] Usage: {0} <best PF> <moea PF> <num. exec.> <computed PF> <num. exec.> <min. coverage>".format(sys.argv[0]))
        exit(-1)

    best_pf_file = sys.argv[1]
    moea_pf_file = sys.argv[2]
    moea_num_exec = int(sys.argv[3])
    comp_pf_file = sys.argv[4]
    comp_num_exec = int(sys.argv[5])
    min_cover = float(sys.argv[6])

    print("Best PF file    : {0}".format(best_pf_file))
    print("MOEA PF file    : {0} ({1})".format(moea_pf_file, moea_num_exec))
    print("Computed PF file: {0} ({1})".format(comp_pf_file, comp_num_exec))
    print("Min. coverage   : {0}".format(min_cover))
    print()

    best_pf = []
    with open(best_pf_file) as f:
        for line in f:
            if len(line.strip()) > 0:
                data = line.strip().split(" ")
                assert(len(data)==3)

                if float(data[1]) >= min_cover:
                    best_pf.append((float(data[0]),1/float(data[1]),float(data[2])))

    maxValues = getMaxValues(best_pf)
    minValues = getMinValues(best_pf)

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
                            moea_pf_exec.append((energy,1/coverage,nforwardings))
                                
        moea_pf.append(moea_pf_exec)
        
        normalized_pf = getNormalizedFront(moea_pf_exec, maxValues, minValues)
        inverted_pf = invertedFront(normalized_pf)
        hv = hypervolume(inverted_pf,len(inverted_pf), len(inverted_pf[0]))
        
        moea_pf_value.append(hv)

    comp_pf = []
    comp_pf_value = []
    for e in range(comp_num_exec):
        comp_pf_exec = []

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
                                comp_pf_exec.append((energy,1/coverage,nforwardings))
                                
        comp_pf.append(comp_pf_exec)
        
        normalized_pf = getNormalizedFront(comp_pf_exec, maxValues, minValues)
        inverted_pf = invertedFront(normalized_pf)
        hv = hypervolume(inverted_pf,len(inverted_pf), len(inverted_pf[0]))
        
        comp_pf_value.append(hv)

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

    (comp_avg, comp_stdev) = avg_stdev(comp_pf_value)
    (moea_avg, moea_stdev) = avg_stdev(moea_pf_value)
    (hstatic, pvalue) = mstats.kruskalwallis(moea_pf_value, comp_pf_value)

    print("alg|avg|stdev")
    print("comp|{0}|{1}".format(comp_avg, comp_stdev))
    print("moea|{0}|{1}".format(moea_avg, moea_stdev))
    print("h-static {0}|p-value {1}".format(hstatic, pvalue))
    
    #0.05	0.01	0.001
    if pvalue <= 0.001: print("1x10^-3")
    elif pvalue <= 0.01: print("1x10^-2")
    elif pvalue <= 0.05: print("5x10^-2")
    
    moea_total = 0
    for e in moea_pf:
        moea_total = moea_total + len(e)
    
    comp_total = 0
    for e in comp_pf:
        comp_total = comp_total + len(e)
        
    print("count moea={0} comp={1}".format(moea_total/len(moea_pf), comp_total/len(comp_pf)))

    return 0

if __name__ == '__main__':
    main()

