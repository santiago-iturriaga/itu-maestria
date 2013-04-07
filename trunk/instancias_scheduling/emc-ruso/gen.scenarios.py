#encoding: utf-8
'''
Created on Oct 3, 2011

@author: santiago
'''

import sys
import random

if __name__ == '__main__':
    argc = len(sys.argv)
    
    if argc != 3:
        print "Modo de uso: python %s <cant_maquinas> <seed>" % sys.argv[0]
        exit(0)

    cantidad_maquinas = int(sys.argv[1])
    current_seed = int(sys.argv[2])
    
    random.seed(current_seed)
    
    lista_proc = open('lista_proc')
    contenido = lista_proc.readlines()
    
    total_maquinas = len(contenido)
    #print "Total máquinas %s" % total_maquinas
    
    for i in range(cantidad_maquinas):
        sorteada = random.randint(0, total_maquinas-1) 
        #print "Sorteada %s" % sorteada
        print contenido[sorteada],
