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
import random
import math

def main(cant_maquinas, cant_scenarios):
    for s in range(cant_scenarios):
        cores_list = []
        
        for maquina in range(cant_maquinas):
            # Calculo la prioridad asignada a la tarea.
            cores_exp = int(random.expovariate(2)) # 0.90
            cores = int(math.pow(2,cores_exp))
            cores_list.append(cores)
            
        cores_max = max(cores_list)
        with open('cores_c' + str(cores_max) + '.' + str(s), 'w') as output:
            for c in cores_list:
                output.write(str(c) + "\n")
        
    return 0

if __name__ == '__main__':
    argc = len(sys.argv)
    
    if argc != 4:
        print("Modo de uso: python {0} <#maquinas> <#scenarios> <seed>".format(sys.argv[0]))
        sys.exit(0)

    cant_maquinas = int(sys.argv[1])
    cant_scenarios = int(sys.argv[2])
    current_seed = int(sys.argv[3])
    
    random.seed(current_seed)

    main(cant_maquinas, cant_scenarios)
