#encoding: utf-8

import os
import math

deterministas = ['MinMin', 'MINMin', 'MinMIN', 'MINMIN']
deterministas_dir = 'list-heuristics/512x16'
no_deterministas_dir = '512x16.test'

def calcular_medidas(valores):
    min = valores[0]
    max = valores[0]
    total = 0.0

    for valor in valores:
        total += valor
        if valor < min: min = valor
        if valor > max: max = valor

    avg = float(total) / float(len(valores))
    
    aux_valor = 0.0
    for valor in valores:
        aux_valor += math.pow(valor-avg,2)

    stddev = math.sqrt(aux_valor / (len(valores)-1))

    return (min, max, avg, stddev)

if __name__ == '__main__':
    instancias_raw = []

    for filename in os.listdir(no_deterministas_dir):
        instancias_raw.append(filename)

    ejecuciones_pals_0 = {}
    ejecuciones_pals_1 = {}

    for instancia in instancias_raw:
        for filename in os.listdir(no_deterministas_dir + '/' + instancia):
            #pals.0.s10.w1.err
            filename_parts = filename.split('.')
            
            if len(filename_parts) == 5:
                file_inst = filename_parts[2] + '.' + filename_parts[3]
                file_version = filename_parts[0] + '.' + filename_parts[1]
                file_type = filename_parts[4]
                file_id = filename_parts[0] + '.' + filename_parts[1] + '.' + filename_parts[2] + '.' + filename_parts[3]
                
                if file_version == 'pals.0':                
                    if not file_version in ejecuciones_pals_0:
                        ejecuciones_pals_0[file_inst] = []
                    ejecuciones_pals_0[file_inst].append(file_id)
                elif file_version == 'pals.1':
                    if not file_version in ejecuciones_pals_1:
                        ejecuciones_pals_1[file_inst] = []
                    ejecuciones_pals_1[file_inst].append(file_id)
                             
    instancias = list(set(instancias_raw))
    instancias.sort()

    #print instancias

    resultados_deterministas = []
    resultados_pals_0 = {}
    resultados_pals_1 = {}

    for instancia in instancias:
        for d in deterministas:
            path = deterministas_dir + '/' + d + '.' + instancia + '.metrics'

            if os.path.isfile(path):
                metrics_file = open(path)
                values = metrics_file.readline().split(' ')
                makespan = float(values[0])
                energy = float(values[1])

                resultados_deterministas.append((instancia, d, makespan, energy))
            else:
                exit(-1)

    for instancia in instancias:
        ejecucion = ejecuciones_pals_0[instancia][0]
        
        path = no_deterministas_dir + '/' + instancia + '/' + ejecucion + '.metrics'
        print path

        if os.path.isfile(path):
            metrics_file = open(path)
            values = metrics_file.readline().split(' ')
            makespan = float(values[0])
            energy = float(values[1])

            resultados_pals_0[instancia] = (ejecucion, makespan, energy)
        else:
            exit(-1)
            
        ejecucion = ejecuciones_pals_1[instancia][0]
        
        path = no_deterministas_dir + '/' + instancia + '/' + ejecucion + '.metrics'
        print path

        if os.path.isfile(path):
            metrics_file = open(path)
            values = metrics_file.readline().split(' ')
            makespan = float(values[0])
            energy = float(values[1])

            resultados_pals_1[instancia] = (ejecucion, makespan, energy)
        else:
            exit(-1)

    print "====== Tabla de makespan ======"
    print "Instancia,MinMin,PALS 1p,PALS 1p vs MinMin,PALS 2p,PALS 2p vs MinMin"
    for instancia in instancias:
        makespan_minmin = 0.0
    
        for (r_instancia, r_d, r_makespan, r_energy) in resultados_deterministas:
            if instancia == r_instancia and r_d == 'MinMin':
                makespan_minmin = r_makespan

        print "%s,%.1f,%.1f,%.1f,%.1f,%.1f" % (instancia, makespan_minmin, \
            resultados_pals_1[instancia][1], 100.0 - (resultados_pals_1[instancia][1] * 100.0 / makespan_minmin), \
            resultados_pals_0[instancia][1], 100.0 - (resultados_pals_0[instancia][1] * 100.0 / makespan_minmin))

    print "====== Tabla de energía ======"
    print "Instancia,MinMin,PALS 1p,PALS 1p vs MinMin,PALS 2p,PALS 2p vs MinMin"
    for instancia in instancias:
        energy_minmin = 0.0
    
        for (r_instancia, r_d, r_makespan, r_energy) in resultados_deterministas:
            if instancia == r_instancia and r_d == 'MinMin':
                energy_minmin = r_energy

        print "%s,%.1f,%.1f,%.1f,%.1f,%.1f" % (instancia, energy_minmin, \
            resultados_pals_1[instancia][2], 100.0 - (resultados_pals_1[instancia][2] * 100.0 / energy_minmin), \
            resultados_pals_0[instancia][2], 100.0 - (resultados_pals_0[instancia][2] * 100.0 / energy_minmin))
