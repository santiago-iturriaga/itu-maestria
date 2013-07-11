#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
#  time_eval.py
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
        print("ERROR. {0} <base_path> <#exec>".format(sys.argv[0]))
        exit(-1)

    nro_exec = int(sys.argv[2])
    base_path = sys.argv[1]

    times_sum = 0.0
    times = []

    for e in range(nro_exec):
        with open(base_path + "." + str(e) + ".time") as f:
            for l in f:
                if l.startswith("real"):
                    raw = l[5:].strip().strip("s")
                    raw_data = raw.split("m")
                    time = float(raw_data[0]) + (float(raw_data[1])/60)
                    times.append(time)
                    times_sum = times_sum + time
                    
    print("average: {0:.2f} m".format(times_sum/len(times)))

    return 0

if __name__ == '__main__':
    main()

