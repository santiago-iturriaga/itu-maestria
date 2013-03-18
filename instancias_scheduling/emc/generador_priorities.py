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
    
    if argc != 3:
        print "Modo de uso: python %s <#tareas> <seed>" % sys.argv[0]
        exit(0)

    cantidad_tareas = int(sys.argv[1])
    current_seed = int(sys.argv[2])
    
    random.seed(current_seed)
    
    for task in range(cantidad_tareas):
        # Calculo la prioridad asignada a la tarea.
        prioridad = random.gauss(5, 2.5)
        
        if prioridad < 1: prioridad = 1
        if prioridad > 10: prioridad = 10
        
        if prioridad >= 5:
            print int(math.floor(prioridad))
        else:
            print int(math.ceil(prioridad))
