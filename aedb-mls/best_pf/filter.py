#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  convert.py
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

def main():
    if len(sys.argv) != 3:
        print("ERROR! {0} <best pf> <min. coverage>".format(sys.argv[0]))
        sys.exit(-1)
        
    best_pf = sys.argv[1]
    min_coverage = float(sys.argv[2])
    
    pf = []
    
    with open(best_pf) as f:
        for line in f:
            data_raw = line.strip().split(" ")
            
            if len(data_raw) == 3:
                energy = float(data_raw[0])
                coverage = float(data_raw[1])
                nforwardings = float(data_raw[2])
                
                if coverage <= -1*min_coverage and energy > 0:
                    pf.append((energy,coverage,nforwardings))
    
    for s in pf:
        print("{0:.4f} {1:.4f} {2:.4f}".format(s[0],s[1],s[2]))
    
    return 0

if __name__ == '__main__':
    main()

