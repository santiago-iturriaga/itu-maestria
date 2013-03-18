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
import random
import numpy

def main(machine_count, scenario_count):
    machines_list = []
    
    low = 0
    mid = 0
    high = 0
    
    with open('lista_proc') as machines_file:
        for m in machines_file:
            machines_list.append(m)
    
    for s in range(scenario_count):
        ssj_values = []
        cores_values = []
        selected_machines = []
        
        for i in range(machine_count):
            m = random.choice(machines_list)
            
            m_data = m.split('\t')
            ssj_values.append(float(m_data[1].strip()) / int(m_data[0].strip()))
            cores_values.append(int(m_data[0].strip()))
            
            selected_machines.append(m)
        
        print ssj_values
        min_ssj = min(ssj_values)
        max_ssj = max(ssj_values)
        #mean_ssj = numpy.mean(ssj_values)
        #stdev_ssj = numpy.std(ssj_values)
        
        print min_ssj
        print max_ssj
        #print mean_ssj
        #print stdev_ssj
        
        ratio = min_ssj / max_ssj
        print "ratio: %.2f" % ratio
        
        print cores_values
        max_cores = max(cores_values)
        print max_cores
        
        if ratio <= 0.33:
            filename = 'scenario_c' + str(max_cores) + '_low.' + str(low)
            low = low + 1
        elif ratio <= 0.66:
            filename = 'scenario_c' + str(max_cores) + '_mid.' + str(mid)
            mid = mid + 1
        else:
            filename = 'scenario_c' + str(max_cores) + '_high.' + str(high)
            high = high + 1
        
        with open(filename, 'w') as output:
            for m in selected_machines:
                output.write(m)
    
    return 0

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Error!"
        print "Usage: python %s <#machines> <#scenarios> <seed>" % sys.argv[0]
        exit(-1)
        
    print "# machines: %s" % sys.argv[1]
    machine_count = int(sys.argv[1])

    print "# scenarios: %s" % sys.argv[2]
    scenario_count = int(sys.argv[2])
    
    print "seed: %s" % sys.argv[3]
    seed = int(sys.argv[3])
        
    random.seed(seed)
    main(machine_count, scenario_count)
