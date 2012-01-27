#encoding: utf-8

import os
import math

list_heur = ['MinMin', 'MINMin', 'MinMIN', 'MINMIN']
list_heur_dir = 'list-heuristics/512x16'
palsRuso_dir = 'pals-ruso/512x16'
pals_dir = '512x16.test'

if __name__ == '__main__':
    instancias_raw = []

    for filename in os.listdir(pals_dir):
        nameParts = filename.split('.')
        instancias_raw.append(nameParts[0] + '.' + nameParts[1])

    ejecuciones_pals = {}

    for instancia in instancias_raw:
        for filename in os.listdir(pals_dir + '/' + instancia):
            filename_parts = filename.split('.')
            
            if len(filename_parts) == 5:
                file_inst = filename_parts[2] + '.' + filename_parts[3]
                file_version = filename_parts[0] + '.' + filename_parts[1]
                file_type = filename_parts[4]
                file_id = filename_parts[0] + '.' + filename_parts[1] + '.' + filename_parts[2] + '.' + filename_parts[3]
                
                if not file_version in ejecuciones_pals:
                    ejecuciones_pals[file_inst] = []
                    ejecuciones_pals[file_inst].append(file_id)
                             
    instancias = list(set(instancias_raw))
    instancias.sort()

    resultados_list_heur = []
    resultados_pals = {}
    resultados_palsRuso = {}

    for instancia in instancias:
        for d in list_heur:
            path = list_heur_dir + '/' + d + '.' + instancia + '.metrics'
            print path
            
            if os.path.isfile(path):
                metrics_file = open(path)
                values = metrics_file.readline().split(' ')
                makespan = float(values[0])
                energy = float(values[1])

                resultados_list_heur.append((instancia, d, makespan, energy))
            else:
                print "[ERROR] cargando heuristica %s" % d
                exit(-1)

    for instancia in instancias:           
        path = palsRuso_dir + '/palsRuso.' + instancia + '.metrics'
        print path
        
        if os.path.isfile(path):
            metrics_file = open(path)
            values = metrics_file.readline().split(' ')
            makespan = float(values[0])
            energy = float(values[1])

            resultados_palsRuso[instancia] = (makespan, energy)
        else:
            print "[ERROR] cargando heuristica palsRuso"
            exit(-1)

    for instancia in instancias:           
        ejecucion = ejecuciones_pals[instancia][0]
        
        path = pals_dir + '/' + instancia + '/' + ejecucion + '.metrics'
        print path

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

            resultados_pals[instancia] = (min_makespan, min_energy)
        else:
            print "[ERROR] cargando heuristica pals"
            exit(-1)
            
    print "====== Tabla de makespan ======"
    print "Instancia,MinMin,PALS Ruso,PALS Ruso vs MinMin,PALS 2obj,PALS 2obj vs MinMin"
    for instancia in instancias:
        makespan_minmin = 0.0
    
        for (r_instancia, r_d, r_makespan, r_energy) in resultados_list_heur:
            if instancia == r_instancia and r_d == 'MinMin':
                makespan_minmin = r_makespan

        print "%s,%.1f,%.1f,%.1f,%.1f,%.1f" % (instancia, makespan_minmin, \
            resultados_palsRuso[instancia][0], 100.0 - (resultados_palsRuso[instancia][0] * 100.0 / makespan_minmin), \
            resultados_pals[instancia][0], 100.0 - (resultados_pals[instancia][0] * 100.0 / makespan_minmin))

    print "====== Tabla de energía ======"
    print "Instancia,MinMin,PALS Ruso,PALS Ruso vs MinMin,PALS 2obj,PALS 2obj vs MinMin"
    for instancia in instancias:
        energy_minmin = 0.0
    
        for (r_instancia, r_d, r_makespan, r_energy) in resultados_list_heur:
            if instancia == r_instancia and r_d == 'MinMin':
                energy_minmin = r_energy

        print "%s,%.1f,%.1f,%.1f,%.1f,%.1f" % (instancia, energy_minmin, \
            resultados_palsRuso[instancia][1], 100.0 - (resultados_palsRuso[instancia][1] * 100.0 / energy_minmin), \
            resultados_pals[instancia][1], 100.0 - (resultados_pals[instancia][1] * 100.0 / energy_minmin))
