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
import numpy

def main(tasks_count, arrival_rate, scenarios_count):
    for s in range(scenarios_count):
        with open('arrival.' + str(s), 'w') as output:
            current_time = 0
            tasks_left = tasks_count
            while tasks_left > 0:
                tasks_arrived = numpy.random.poisson(arrival_rate)
                
                if (tasks_arrived > 0):
                    if (tasks_arrived <= tasks_left):
                        output.write(str(current_time) + "\t" + str(tasks_arrived) + "\n")
                    else:
                        output.write(str(current_time) + "\t" + str(tasks_left) + "\n")
                    
                tasks_left = tasks_left - tasks_arrived
                current_time = current_time + 1
    
            print("> tasks arrived in {0:d} seconds ({1:d} minutes, or {2:d} hours)\n".format(current_time, current_time / 60, (current_time / 60) / 60))
    return 0

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python {0} <#tasks> <arrival rate/s> <#scenarios> <seed>".format(sys.argv[0]))
        exit(-1)
    
    print("# tasks       : {0}" % sys.argv[1]
    print("arrival rate/s: {0}" % sys.argv[2]
    print("# scenarios   : {0}" % sys.argv[3]
    print("seed          : {0}" % sys.argv[4]
    
    tasks_count = int(sys.argv[1])
    arrival_rate = float(sys.argv[2])
    scenarios_count = int(sys.argv[3])
    seed = int(sys.argv[4])
    
    numpy.random.seed(seed)
    
    main(tasks_count, arrival_rate, scenarios_count)
