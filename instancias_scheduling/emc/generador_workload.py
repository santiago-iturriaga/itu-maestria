#encoding: utf-8
'''
Created on Oct 3, 2011

@author: santiago
'''

import sys
import random

# Todas las unidades de tiempo son segundos.
TO_min = 0.25*60*60 # 15 minutos.
TO_max = 72*60*60   # 72 horas.

# Intel Xeon E5440: cores=4, ssj_ops=150,979, E_IDLE=76.9, E_MAX=131.8
TO_default_ssj = float(150979 / 4)
TO_min_ssj = float(TO_default_ssj) * float(TO_min)
TO_max_ssj = float(TO_default_ssj) * float(TO_max) 

Heter_step = (TO_max_ssj - TO_min_ssj) / 5
Heter_mult = [1, 3, 5]

if __name__ == '__main__':
    argc = len(sys.argv)
    
    if argc != 6:
        print "Modo de uso: python %s <#tareas> <#maquinas> <heterogeneidad> <#scenarios> <seed>" % sys.argv[0]
        print "             heterogeneidad: LOW=1, MEDIUM=2, HIGH=3"
        exit(0)

    cantidad_tareas = int(sys.argv[1])
    cantidad_maquinas = int(sys.argv[2])
    heterogeneidad = int(sys.argv[3])
    cantidad_scenarios = int(sys.argv[4])
    current_seed = int(sys.argv[5])
       
    random.seed(current_seed)
    
    print "cantidad tareas: %d" % cantidad_tareas
    print "cantidad maquinas: %d" % cantidad_maquinas
    print "heterogeneidad: %d" % heterogeneidad
    print "cantidad scenarios: %d" % cantidad_scenarios
    print "seed: %d" % current_seed
    print "min: %d" % TO_min_ssj
    print "max: %d" % TO_max_ssj
    
    if heterogeneidad == 1: filename = 'workload_low.'
    elif heterogeneidad == 2: filename = 'workload_mid.'
    else: filename = 'workload_high.'
    
    for s in range(cantidad_scenarios):
        with open(filename + str(s), 'w') as output:
            for task in range(cantidad_tareas):
                # Calculo el costo independiente de la máquina.
                current_heter_mult = Heter_mult[random.randint(1,heterogeneidad) - 1]
                TO_current = long(random.uniform(TO_min_ssj, TO_min_ssj + Heter_step) * current_heter_mult)
                
                #for machine in range(cantidad_maquinas):
                    # Calculo el costo del overhead adicional para cada posible máquina.
                    #if heterogeneidad == 0:
                    #    AO_current = 0
                    #else:
                    #    AO_current = random.randint(AO_hetero[0], AO_hetero[1]) 
                    #print(AO_current)

                    # Calculo TO * (1 + AO).        
                    #print long(TO_current * ((AO_current / 100.0) + 1))
                    
                output.write(str(TO_current) + "\n")
