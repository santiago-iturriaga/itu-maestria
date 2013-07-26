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
import array

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

def main():
    if len(sys.argv) != 5:
        print("[ERROR] Usage: {0} <true PF> <computed PF> <num. exec.> <min. coverage>".format(sys.argv[0]))
        exit(-1)

    true_pf_file = sys.argv[1]
    comp_pf_file = sys.argv[2]
    num_exec = int(sys.argv[3])
    min_cover = float(sys.argv[4])

    #print("True PF file    : {0}".format(true_pf_file))
    #print("Computed PF file: {0}".format(comp_pf_file))
    #print("Num. executions : {0}".format(num_exec))
    #print("Min. coverage   : {0}".format(min_cover))
    #print()

    true_pf = []
    with open(true_pf_file) as f:
        for line in f:
            if len(line.strip()) > 0:
                data = line.strip().split(" ")
                assert(len(data)==3)

                if float(data[1]) >= min_cover:
                    true_pf.append((float(data[0]),1/float(data[1]),float(data[2])))

    maxValues = getMaxValues(true_pf)
    minValues = getMinValues(true_pf)

    #print(maxValues)
    #print(minValues)
    
    hvolumes_sum = 0.0
    hvolumes = []

    for e in range(num_exec):
        comp_pf = []

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
                                comp_pf.append((energy,1/coverage,nforwardings))

        #print("==================================================")
        #print(comp_pf)

        normalized_pf = getNormalizedFront(comp_pf, maxValues, minValues)
        
        #print("==================================================")
        #print(normalized_pf)
        
        inverted_pf = invertedFront(normalized_pf)
        
        #print("==================================================")
        #print(inverted_pf)
        
        hv = hypervolume(inverted_pf,len(inverted_pf), len(inverted_pf[0]))
        
        hvolumes.append(hv)
        hvolumes_sum = hvolumes_sum + hv

    for i in hvolumes:
        print(i)

    #hvolumes_average = hvolumes_sum / len(hvolumes)
    
    #hvolumes_sqsum = 0.0
    #for i in hvolumes: hvolumes_sqsum = hvolumes_sqsum + pow(i-hvolumes_average,2)
    #hvolumes_stdev = math.sqrt(hvolumes_sqsum/(len(hvolumes)-1))

    #print("{0:.4f} {1:.4f}".format(hvolumes_average,hvolumes_stdev))

    return 0

if __name__ == '__main__':
    main()

