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

import os

def main():
    file_id = 0
    dimension = "50"
    
    for dirname, dirnames, filenames in os.walk('.'):
        for filename in filenames:
            if "OAR.aedb."+dimension+"." in filename:
                if ".stdout" in filename:
                    print filename
                    
                    output_content = ""
                    
                    pos = 0
                    for line in open(filename):
                        pos = pos + 1
                        
                        if pos >= 11:
                            data = line.split(",")
                            output_content = output_content + data[6] + "," + data[7] + "," + data[8] + "\n"
                    
                    output_filename = "FUN." + str(file_id)
                    print output_content
                    
                    output_file = open(output_filename, "w")
                    output_file.write(output_content)
                    output_file.close()
                    
                    file_id = file_id + 1

    return 0

if __name__ == '__main__':
    main()
