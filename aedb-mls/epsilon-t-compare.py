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
import subprocess
from scipy.stats import mstats
from scipy import stats

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

    with open("/home/santiago/aux_best.pf","w") as f:
         for l in best_pf:
             f.write("{0} {1} {2}\n".format(l[0],-1*l[1],l[2]))

    print("==== moea...")

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
                                
        with open("/home/santiago/aux_approx.pf","w") as f:
            for l in moea_pf_exec:
                f.write("{0} {1} {2}\n".format(l[0],-1*l[1],l[2]))

        moea_pf.append(moea_pf_exec)

        #p = subprocess.Popen("java -classpath /home/siturria/itu-maestria/trunk/metricas_mo/java/bin jmetal.qualityIndicator.Epsilon /home/siturria/aux_approx.pf /home/siturria/aux_best.pf 3", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p = subprocess.Popen("java -classpath /home/santiago/google-hosting/itu-maestria/svn/trunk/metricas_mo/java/bin jmetal.qualityIndicator.Epsilon /home/santiago/aux_approx.pf /home/santiago/aux_best.pf 3", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        line = p.stdout.readlines()[0]
        #.strip("\n").strip()
        #print(str(line)) 
        epsilon_value = float(line)
        retval = p.wait()

        print(epsilon_value)
        if epsilon_value > 0: exit(-1)

        subprocess.call(["rm","/home/santiago/aux_approx.pf"])

        #moea_pf_value.append(epsilon_metric(best_pf, moea_pf_exec))
        moea_pf_value.append(epsilon_value)

    print("==== computed...")

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
                                comp_pf_exec.append((energy,coverage,nforwardings))
                                
        with open("/home/santiago/aux_comp.pf","w") as f:
            for l in comp_pf_exec:
                f.write("{0} {1} {2}\n".format(l[0],-1*l[1],l[2]))
                                
        comp_pf.append(comp_pf_exec)
        
        #p = subprocess.Popen("java -classpath /home/siturria/itu-maestria/trunk/metricas_mo/java/bin jmetal.qualityIndicator.Epsilon /home/siturria/aux_approx.pf /home/siturria/aux_best.pf 3", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p = subprocess.Popen("cat ", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        line = p.stdout.readlines()[0]
        #.strip("\n").strip()
        #print(str(line)) 
        epsilon_value = float(line)
        retval = p.wait()

        print(epsilon_value)
        if epsilon_value < 0: exit(-1)

        subprocess.call(["rm","/home/santiago/aux_comp.pf"])
        
        #moea_pf_value.append(epsilon_metric(best_pf, moea_pf_exec))
        comp_pf_value.append(epsilon_value)

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

    #print(comp_pf_value)
    #print(moea_pf_value)

    (comp_avg, comp_stdev) = avg_stdev(comp_pf_value)
    (moea_avg, moea_stdev) = avg_stdev(moea_pf_value)
    (hstatic, pvalue) = mstats.kruskalwallis(moea_pf_value, comp_pf_value)
    #(u, prob) = stats.mannwhitneyu(moea_pf_value, comp_pf_value)

    print("alg|avg|stdev")
    print("comp|{0}|{1}".format(comp_avg, comp_stdev))
    print("moea|{0}|{1}".format(moea_avg, moea_stdev))
    print("h-static {0}|p-value {1}".format(hstatic, pvalue))
    #print("u {0}|prob {1}".format(u, prob))
    
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

