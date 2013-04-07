#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  untitled.py
#  
#  Copyright 2013 Santiago Iturriaga <santiago@marga>
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

def get_utopic_point(pf):
    sorted_pf = sorted(pf)
    return (sorted_pf[0][0], sorted_pf[-1][1])

def get_zenith_point(pf):
    sorted_pf = sorted(pf)
    return (sorted_pf[-1][0], sorted_pf[0][1])

def euclidean_distance(a, b):
    return math.sqrt(pow(a[0]-b[0],2)+pow(a[1]-b[1],2))

def closest_point(a, list_b):
    closest_b = list_b[0]
    closest_b_distance = euclidean_distance(a,list_b[0])
    
    for b in list_b:
        distance = euclidean_distance(a,b)
        
        if distance < closest_b_distance:
            closest_b = b
            closest_b_distance = distance
            
    return (closest_b, closest_b_distance)

def igd(pf, approx_pf):
    partial_sum = 0
    
    for pf_s in pf:
        d = closest_point(pf_s, approx_pf)
        partial_sum = pow(d[1],2)
        
    return math.sqrt(partial_sum) / len(pf)

def batch_igd(dimension):
    INSTANCES = 30
    SCENARIOS = (0,3,6,9,10,11,13,14,16,17,19)
    WORKLOADS = sorted(("A.u_c_hihi","A.u_c_hilo","A.u_c_lohi","A.u_c_lolo","A.u_i_hihi","A.u_i_hilo","A.u_i_lohi","A.u_i_lolo","A.u_s_hihi","A.u_s_hilo","A.u_s_lohi","A.u_s_lolo","B.u_c_hihi","B.u_c_hilo","B.u_c_lohi","B.u_c_lolo","B.u_i_hihi","B.u_i_hilo","B.u_i_lohi","B.u_i_lolo","B.u_s_hihi","B.u_s_hilo","B.u_s_lohi","B.u_s_lolo"))
    
    #INSTANCES = 30
    #SCENARIOS = (0,)
    #WORKLOADS = ("A.u_i_hihi",)

    for scenario in SCENARIOS:
        for workload in WORKLOADS:
            pf_file = "../"+dimension+".fp/scenario."+str(scenario)+".workload."+workload+".fp"
            
            pf = []
            with open(pf_file) as f:
                for line in f:
                    values = line.strip().split(' ')
                    pf.append((float(values[0]), float(values[1])))
            
            u = get_utopic_point(pf)
            z1 = get_zenith_point(pf)
            
            #norm_pf = []
            #for s in pf:
            #    norm_pf.append((s[0]/u[0], s[1]/u[1]))
            
            path_approx_pf = ["../"+dimension+".24_10s","../"+dimension+".24.adhoc"]
            preffix_approx_pf = ["pals-aga","pals-1"]
            
            out_path_approx_pf = [dimension+"/metrics-aga",dimension+"/metrics-adhoc"]
            out_preffix_approx_pf = ["pals-aga","pals-1"]
            
            #path_approx_pf = ["../"+dimension+".24_10s"]
            #preffix_approx_pf = ["pals-aga"]
            
            #out_path_approx_pf = [dimension+"/metrics-aga"]
            #out_preffix_approx_pf = ["pals-aga"]
            
            for execution in range(len(path_approx_pf)):
                for instance in range(INSTANCES):
                    approx_pf_file = path_approx_pf[execution]+"/scenario."+str(scenario)+".workload."+workload+"."+str(instance)+"/"+preffix_approx_pf[execution]+".scenario."+str(scenario)+".workload."+workload+".metrics"
                    
                    approx_pf = []
                    with open(approx_pf_file) as f:
                        for line in f:
                            values = line.strip().split(' ')
                            approx_pf.append((float(values[0]), float(values[1])))
                    
                    z2 = get_zenith_point(approx_pf)
                    z = [0,0]
                    z[0] = max(z1[0],z2[0])
                    z[1] = max(z1[1],z2[1])
                    
                    norm_pf = []
                    for s in pf:
                        if z[0] != u[0]:
                            #norm_pf.append((s[0]/u[0], s[1]/u[1]))
                            norm_pf.append(((s[0]-u[0])/(z[0]-u[0]), (s[1]-u[1])/(z[1]-u[1])))
                        else:
                            norm_pf.append((s[0]/u[0], s[1]/u[1]))
                            
                    norm_approx_pf = []
                    for s in approx_pf:
                        if z[0] != u[0]:
                            #norm_approx_pf.append((s[0]/u[0], s[1]/u[1]))
                            norm_approx_pf.append(((s[0]-u[0])/(z[0]-u[0]), (s[1]-u[1])/(z[1]-u[1])))
                        else:
                            norm_approx_pf.append((s[0]/u[0], s[1]/u[1]))

                    #print u
                    #print z
                    #print pf
                    #print approx_pf
                    #print norm_pf
                    #print norm_approx_pf

                    v = igd(norm_pf, norm_approx_pf)
                    
                    #print v
                    #print "=============================================="
                    
                    out_file = out_path_approx_pf[execution]+"/"+out_preffix_approx_pf[execution]+".scenario."+str(scenario)+".workload."+workload+"."+str(instance)+".igd"
                    print out_file
                    print v
                    with open(out_file, "w") as o:
                        o.write(str(v))
                        
                    #print "%s %s %s = %s" % (scenario, workload,instance,v)

def main():
    batch_igd("512x16")
    #batch_igd("1024x32")
    #batch_igd("2048x64")
    
    return 0

if __name__ == '__main__':
    main()
