#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  procesar.py
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
import io
import math

def main():
    TASKS=(8192,16384,32768)
    MACHINES=(256,512,1024)
    
    INSTANCES=20
    
    for d in range(3):
        print("{0}x{1}".format(TASKS[d],MACHINES[d]))
        
        for i in range(INSTANCES):
            times=[]
            
            with open("result-2/etc_c_{0}x{1}_hihi-{2}.log".format(TASKS[d],MACHINES[d],i+1)) as f:
                for line in f:
                    if len(line.strip()) > 0:
                        times.append(float(line.strip()))
                        
            sum_times = 0.0
            for t in times:
                sum_times = sum_times + t
                
            avg_times = sum_times / len(times)
            
            sq_times = 0.0
            for t in times:
                sq_times = sq_times + pow(t-avg_times,2)
                
            stdev_times = math.sqrt(sq_times / (len(times)-1))
            
            #print("{0} {1:.2f} {2:.2f}".format(i+1, avg_times, stdev_times))
            print("{0} {1:.2f} {2:.2f}".format(i+1, avg_times / 1000000, stdev_times / 1000000))
    
    return 0

if __name__ == '__main__':
    main()

