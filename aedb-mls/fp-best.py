#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  fp-eval.py
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
import array

def dominancia(sol_a, sol_b):
    assert(len(sol_a)==len(sol_b))

    if sol_a[0] <= sol_b[0] and sol_a[1] >= sol_b[1] and sol_a[2] <= sol_b[2]:
        return 1
    elif sol_a[0] >= sol_b[0] and sol_a[1] <= sol_b[1] and sol_a[2] >= sol_b[2]:
        return -1
    else:
        return 0

def main():
    if len(sys.argv) != 3:
        print("[ERROR] Usage: {0} <mls PF> <moea PF>".format(sys.argv[0]))
        exit(-1)

    mls_pf_file = sys.argv[1]
    moea_pf_file = sys.argv[2]

    #print("MLS PF file: {0}".format(mls_pf_file))
    #print("MOEA PF file: {0}".format(moea_pf_file))

    pf_final = []

    with open(mls_pf_file) as f:
        for line in f:
            if len(line.strip()) > 0:
                data = line.strip().split(" ")

                if len(data) == 3:
                    if (data[0] != "id"):
                        energy = float(data[0])
                        coverage = float(data[1])
                        nforwardings = float(data[2])

                        pf_final.append((energy,coverage,nforwardings))

    with open(moea_pf_file) as f:
        for line in f:
            if len(line.strip()) > 0:
                data = line.strip().split(" ")

                if len(data) == 3:
                    if (data[0] != "id"):
                        energy = float(data[0])
                        coverage = float(data[1])
                        nforwardings = float(data[2])

                        pf_final.append((energy,coverage,nforwardings))

    global_pf = []
    domination=array.array('i',(0,)*1000)

    for i in range(len(pf_final)):
        j = i+1

        while domination[i]==0 and j<len(pf_final):
            result = dominancia(pf_final[i], pf_final[j])

            if result == 0:
                # Ninguno es dominado por el otro
                pass
            elif result == -1:
                # El primero es dominado
                domination[i] = -1;
            elif result == 1:
                # El primero domina
                domination[j] = -1;

            j = j+1

        if domination[i]==0:
            global_pf.append(pf_final[i])

    for i in global_pf: print("{0:.4f} {1:.4f} {2:.4f}".format(i[0],i[1],i[2]))

    return 0

if __name__ == '__main__':
    main()

