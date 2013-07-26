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
    if len(sys.argv) != 3:
        print("[ERROR] Usage: {0} <computed PF> <min. coverage>".format(sys.argv[0]))
        exit(-1)

    comp_pf_file = sys.argv[1]
    min_cover = float(sys.argv[2])

    #print("Computed PF file: {0}".format(comp_pf_file))
    #print("Num. executions : {0}".format(num_exec))
    #print("Min. coverage   : {0}".format(min_cover))
    #print()

    with open(comp_pf_file) as f:
        for line in f:
            if len(line.strip()) > 0:
                data = line.strip().split("\t")
                assert(len(data)==3)

                if -1*float(data[1]) >= min_cover:
                    print("{0} {1} {2}".format(float(data[0]),1/(-1*float(data[1])),float(data[2])))
    
    return 0

if __name__ == '__main__':
    main()

