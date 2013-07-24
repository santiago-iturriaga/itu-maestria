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

def main():
    if len(sys.argv) != 5:
        print("[ERROR] Usage: {0} <computed PF> <num. exec.> <min. coverage> <output>".format(sys.argv[0]))
        exit(-1)

    comp_pf_file = sys.argv[1]
    num_exec = int(sys.argv[2])
    min_cover = float(sys.argv[3])
    output_file = sys.argv[4]

    print("Computed PF file: {0}".format(comp_pf_file))
    print("Num. executions : {0}".format(num_exec))
    print("Min. coverage   : {0}".format(min_cover))
    print("Output file     : {0}".format(output_file))
    print()

    cant = 0

    for e in range(num_exec):
        with open(comp_pf_file + "." + str(e) + ".err") as f:
            print(comp_pf_file + "." + str(e) + ".err")
            line = f.readline()

            while line:
                if line.startswith("[POPULATION]"):
                    data = line.strip().split("=")
                    assert(len(data)==2)

                    count = int(data[1])

                    with open(output_file + "." + str(e) + "." + str(cant), "w") as o:
                        for i in range(count):
                            line = f.readline()
                            data = line.strip().split(",")

                            energy = float(data[-4])
                            coverage = float(data[-3])
                            nforwardings = float(data[-2])

                            if coverage > min_cover:
                                o.write("{0} {1} {2}\n".format(energy,coverage,nforwardings))
                                
                        cant = cant + 1

                line = f.readline()

        with open(comp_pf_file + "." + str(e) + ".out") as f:
            with open(output_file + "." + str(e) + "." + str(cant), "w") as o:
                for line in f:
                    if len(line.strip()) > 0:
                        data = line.strip().split(",")

                        if len(data) == 10:
                            if (data[0] != "id"):
                                energy = float(data[-4])
                                coverage = float(data[-3])
                                nforwardings = float(data[-2])

                                if coverage > min_cover:
                                    o.write("{0} {1} {2}\n".format(energy,coverage,nforwardings))
                                    
        cant = 0

    return 0

if __name__ == '__main__':
    main()

