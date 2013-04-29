#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  avg_time.py
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

import math

def main():
    rel_path = "OPIStudy/data/NSGAII_OPI/FE-HCSP_"
    dimensions = ("8x2","16x3","32x4","512x16")
    instances = (1,2,3)
    num_exec = 5

    for d in dimensions:
        for i in instances:
            all_min_wft = []
            all_min_nrg = []

            for n in range(num_exec):
                min_wft = float('inf')
                min_nrg = float('inf')
                
                with open(rel_path + d + "_" + str(i) + "/FUN." + str(n)) as f:
                    for line in f:
                        data_array = line.strip().split(' ')
                        
                        if (len(data_array) == 2):
                            wft = float(data_array[0])
                            nrg = float(data_array[1])
                            
                            if (min_wft > wft): min_wft = wft
                            if (min_nrg > nrg): min_nrg = nrg
                        
                all_min_wft.append(min_wft)
                all_min_nrg.append(min_nrg)
                               
            mean_wft = sum(all_min_wft) / num_exec
            d_wft = [ (j - mean_wft) ** 2 for j in all_min_wft]
            std_dev_wft = math.sqrt(sum(d_wft) / len(d_wft))

            mean_nrg = sum(all_min_nrg) / num_exec
            d_nrg = [ (j - mean_nrg) ** 2 for j in all_min_nrg]
            std_dev_nrg = math.sqrt(sum(d_nrg) / len(d_nrg))
            
            print(d + " " + str(i) + " | wft: {0:.2f} +/- {1:.2f}  | wft: {2:.2f} +/- {3:.2f}".format(mean_wft, std_dev_wft, mean_nrg, std_dev_nrg))

    return 0

if __name__ == '__main__':
    main()

