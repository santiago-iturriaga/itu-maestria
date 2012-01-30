#encoding: utf-8

import os
import math

cant_iters=5
minmin_dir = 'list-heuristics/2048x64'
pals_ruso_dir = 'pals-ruso/2048x64'
pals_dir = '2048x64.test'

if __name__ == '__main__':
    instancias_raw = []

    for filename in os.listdir(pals_dir):
        nameParts = filename.split('.')
        instancias_raw.append((nameParts[1],nameParts[3]+'.'+nameParts[4]))

    instancias = list(set(instancias_raw))
    instancias.sort()

    resultados_minmin = {}
    resultados_pals = {}
    resultados_pals_ruso = {}

    for instancia in instancias:
        path = minmin_dir + '/MinMin.scenario.' + instancia[0] + '.workload.' + instancia[1] + '.metrics'
        #print path
        
        if os.path.isfile(path):
            metrics_file = open(path)
            values = metrics_file.readline().split(' ')
            makespan = float(values[0])
            energy = float(values[1])

            resultados_minmin[instancia] = (makespan, energy)
        else:
            print "[ERROR] cargando heuristica MinMin"
            exit(-1)

    for instancia in instancias:           
        path = pals_ruso_dir + '/pals-ruso.scenario.' + instancia[0] + '.workload.' + instancia[1] + '.metrics'
        #print path
        
        if os.path.isfile(path):
            metrics_file = open(path)
            values = metrics_file.readline().split(' ')
            makespan = float(values[0])
            energy = float(values[1])

            resultados_pals_ruso[instancia] = (makespan, energy)
        else:
            print "[ERROR] cargando heuristica PALS del Ruso"
            exit(-1)

    for instancia in instancias: 
        abs_min_makespan = 0.0
        abs_min_energy = 0.0
        
        aux_iter_metrics = []
    
        for iter in range(cant_iters):                      
            dir_path = pals_dir + '/scenario.' + instancia[0] + '.workload.' + instancia[1] + '.' + str(iter) + '/'
            file_name = 'pals-1.scenario.' + instancia[0] + '.workload.' + instancia[1] + '.metrics'
            
            path = dir_path + file_name 
            #print path

            if os.path.isfile(path):
                metrics_file = open(path)
                
                min_makespan = 0.0
                min_energy = 0.0
                
                for line in metrics_file:
                    values = line.split(' ')
                    makespan = float(values[0])
                    energy = float(values[1])
                    
                    if min_makespan == 0.0: min_makespan = makespan
                    elif min_makespan > makespan: min_makespan = makespan
                    
                    if min_energy == 0.0: min_energy = energy
                    elif min_energy > energy: min_energy = energy

                if abs_min_makespan == 0.0: abs_min_makespan = min_makespan
                elif abs_min_makespan > min_makespan: abs_min_makespan = min_makespan
                
                if abs_min_energy == 0.0: abs_min_energy = min_energy
                elif abs_min_energy > min_energy: abs_min_energy = min_energy

                aux_iter_metrics.append((min_makespan, min_energy))
            else:
                print "[ERROR] cargando heuristica pals"
                exit(-1)
                
        resultados_pals[instancia] = (abs_min_makespan, abs_min_energy)
            
    print "====== Tabla de makespan ======"
    print "Instancia,MinMin,PALS Ruso,PALS Ruso vs MinMin,PALS 2obj,PALS 2obj vs MinMin"
    for instancia in instancias:
        print "%s,%.1f,%.1f,%.1f,%.1f,%.1f" % ('s' + instancia[0] + ' ' + instancia[1], \
            resultados_minmin[instancia][0], \
            resultados_pals_ruso[instancia][0], 100.0 - (resultados_pals_ruso[instancia][0] * 100.0 / resultados_minmin[instancia][0]), \
            resultados_pals[instancia][0], 100.0 - (resultados_pals[instancia][0] * 100.0 / resultados_minmin[instancia][0]))

    print "====== Tabla de energía ======"
    print "Instancia,MinMin,PALS Ruso,PALS Ruso vs MinMin,PALS 2obj,PALS 2obj vs MinMin"
    for instancia in instancias:
        print "%s,%.1f,%.1f,%.1f,%.1f,%.1f" % ('s' + instancia[0] + ' ' + instancia[1], \
            resultados_minmin[instancia][1], \
            resultados_pals_ruso[instancia][1], 100.0 - (resultados_pals_ruso[instancia][1] * 100.0 / resultados_minmin[instancia][1]), \
            resultados_pals[instancia][1], 100.0 - (resultados_pals[instancia][1] * 100.0 / resultados_minmin[instancia][1]))
