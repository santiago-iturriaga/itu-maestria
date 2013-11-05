# -*- coding: utf-8 -*-
#
#  transform.py
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
    log_name = ""
    
    if len(sys.argv) != 2:
        print("Error: python {0:s} <log file>".format(sys.argv[0]))
        sys.exit(-1)
        
    log_name = sys.argv[1]
    
    with open(log_name) as log_file:
        archivo_line = log_file.readline()
        dimension_line = log_file.readline()
        makespan_line = log_file.readline()
        load_time_line = log_file.readline()
        exec_time_line = log_file.readline()
        
        num_tasks = int(dimension_line.split(",")[0].split(":")[1].strip())
        num_machines = int(dimension_line.split(",")[1].split(":")[1].strip())
        
        makespan = float(makespan_line.split(":")[1].strip())
        
        load_time = int(load_time_line.split(":")[1].strip())
        exec_time = int(exec_time_line.split(":")[1].strip())
    
        assignment_line = log_file.readline()
        assignment = assignment_line.strip()[1:-1].split(" ")
    
        for a in assignment:
            print(a)
    
    return 0

if __name__ == '__main__':
    main()

