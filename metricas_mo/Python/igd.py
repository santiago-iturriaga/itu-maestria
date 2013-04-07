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

def main():
    if len(sys.argv) != 3:
        print "Error!"
        print "Usage: %s <archivo pf> <archivo approx. pf>" % sys.argv[0]
        exit(-1)
    
    pf_file = sys.argv[1]
    approx_pf_file = sys.argv[2]
    
    pf = []
    with open(pf_file) as f:
        for line in f:
            values = line.strip().split(' ')
            pf.append((float(values[0]), float(values[1])))
    
    approx_pf = []
    with open(approx_pf_file) as f:
        for line in f:
            values = line.strip().split(' ')
            approx_pf.append((float(values[0]), float(values[1])))
    
    #print pf
    #print approx_pf
    
    u = get_utopic_point(pf)
    
    norm_pf = []
    for s in pf:
        norm_pf.append((s[0]/u[0], s[1]/u[1]))
        
    norm_approx_pf = []
    for s in approx_pf:
        norm_approx_pf.append((s[0]/u[0], s[1]/u[1]))
    
    #print norm_pf
    #print norm_approx_pf
    
    #print igd(pf, approx_pf)
    print igd(norm_pf, norm_approx_pf)
    
    return 0

if __name__ == '__main__':
    main()
