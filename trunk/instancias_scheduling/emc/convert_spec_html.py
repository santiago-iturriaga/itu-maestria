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
from bs4 import BeautifulSoup, Tag

def main(input_file):
    f = open(input_file)
    content = f.read()

    soup = BeautifulSoup(content)
    for td in soup.find_all('td'):
        if 'hwnodes' in td['class']:
            if td.text == '1':
                cpu_desc = ""
                cores = 0
                ssj_ops = 0
                min_ngr = 0
                max_ngr = 0
                
                #print td.parent
                try:
                    for child in td.parent.descendants:
                        if type(child) == Tag:
                            if child.name == 'td':
                                #print type(child)
                                #print child
                                #print child['class']
                                
                                if 'wkld_ssj_global_config_hw_cpu' in child['class']:
                                    cpu_desc = child.text
                                elif 'aggregate_config_cpu_cores' in child['class']:
                                    cores = int(child.text)
                                elif 'submetric_ssjops_0' in child['class']:
                                    ssj_ops = int(child.text.replace(',','').replace('.',''))
                                elif 'submetric_power_0' in child['class']:
                                    max_ngr = float(child.text.replace(',',''))
                                elif 'submetric_power_10' in child['class']:
                                    min_ngr = float(child.text.replace(',',''))
                            
                    out_string = '%d\t%d\t%.1f\t%.1f\t\t# %s' % (cores, ssj_ops, min_ngr, max_ngr, cpu_desc)
                    print out_string
                except:
                    #print "Omitido..."
                    pass
    
    return 0

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage: python %s <input file>" % sys.argv[0]
        exit(-1)
    
    input_file = sys.argv[1]
    #print "input file: %s" % input_file
    
    main(input_file)
