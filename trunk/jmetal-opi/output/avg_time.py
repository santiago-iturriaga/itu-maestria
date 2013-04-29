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
            time_array_ms = []
            time_total_ms = 0

            for n in range(num_exec):
                with open(rel_path + d + "_" + str(i) + "/Log." + str(n)) as f:
                    l = f.readline()
                    time_ms = int(l.strip())
                    time_total_ms = time_total_ms + time_ms
                    time_array_ms.append(time_ms)

            #print(time_array_ms)

            mean = time_total_ms / num_exec
            dd = [ (j - mean) ** 2 for j in time_array_ms]
            std_dev = math.sqrt(sum(dd) / len(dd))
            
            print(d + " " + str(i) + ": {0:.2f} +/- {1:.2f}".format(mean/1000, std_dev/1000))

    return 0

if __name__ == '__main__':
    main()

