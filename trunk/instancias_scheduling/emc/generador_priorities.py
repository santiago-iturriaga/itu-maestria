#encoding: utf-8
'''
Created on Oct 3, 2011

@author: santiago
'''

import sys
import random
import math

if __name__ == '__main__':
    argc = len(sys.argv)
    
    if argc != 4:
        print("Modo de uso: python {0} <#tareas> <#scenarios> <seed>".format(sys.argv[0]))
        sys.exit(0)

    cantidad_tareas = int(sys.argv[1])
    cantidad_scenarios = int(sys.argv[2])
    current_seed = int(sys.argv[3])
    
    random.seed(current_seed)
    
    for s in range(cantidad_scenarios):
        with open('priorities.' + str(s), 'w') as output:
            for task in range(cantidad_tareas):
                # Calculo la prioridad asignada a la tarea.
                prioridad = random.gauss(3, 1)
                
                if prioridad < 1: prioridad = 1
                if prioridad > 5: prioridad = 5
                
                if prioridad >= 3:
                    output.write(str(int(math.floor(prioridad))) + "\n")
                else:
                    output.write(str(int(math.ceil(prioridad))) + "\n")
