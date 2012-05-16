#encoding: utf-8

import os
import math

cantidad_instancias = 20
algoritmos = ['pminmin', 'minmin', 'pals']

if __name__ == '__main__':
    resultados = {}

    for instancia in range(21)[1:]:
        resultados[instancia] = {}
        
        for a in algoritmos:
            base_path = 'solutions/' + str(instancia) + '.' + a
            print base_path

            if os.path.isfile(base_path + '.makespan'):
                dmake_file = open(base_path + '.makespan')
                dmake = float(dmake_file.readline())

                dtime_file = open(base_path + '.time')
                dtime_lines = dtime_file.readlines()

                if a == 'pals':
                    dtime_line = dtime_lines[5].strip()
                    dtime_str = dtime_line.split('\t')[1].strip()
                    dtime_mins = int(dtime_str.split('m')[0].strip())
                    dtime_secs = float(dtime_str.split('m')[1].strip().strip('s').strip())
                    dtime = dtime_mins * 60 + dtime_secs
                else:
                    dtime_line = dtime_lines[1].strip()
                    dtime_str = dtime_line.split('\t')[1].strip()
                    dtime_mins = int(dtime_str.split('m')[0].strip())
                    dtime_secs = float(dtime_str.split('m')[1].strip().strip('s').strip())
                    dtime = dtime_mins * 60 + dtime_secs

                resultados[instancia][a] = (instancia, a, dmake, dtime)
            else:
                exit(-1)

    print resultados

    print "====== Tabla de makespan ======"
    print "Instancia,MinMin,pMinMin/D,PALS GPU,Improv. pMinMin vs MinMin,Improv. PALS GPU vs MinMin"
    for instancia in range(21)[1:]:
        (minmin_make, minmin_time) = resultados[instancia]['minmin']
        (pminmin_make, pminmin_time) = resultados[instancia]['pminmin']
        (pals_make, pals_time) = resultados[instancia]['pals']
        
        print "%d,%.1f,%.1f,%.1f,%.1f,%.1f" % (str(instancia), \
            minmin_make, pminmin_make, pals_make, \
            100-(pminmin_make * 100 / minmin_make), \
            100-(pals_make * 100 / minmin_make))

    print "====== Tabla de tiempos (segundos) ======"
    print "Instancia,MinMin,pMinMin/D,PALS GPU,Improv. pMinMin vs MinMin,Improv. PALS GPU vs MinMin"
    for instancia in range(21)[1:]:
        (minmin_make, minmin_time) = resultados[instancia]['minmin']
        (pminmin_make, pminmin_time) = resultados[instancia]['pminmin']
        (pals_make, pals_time) = resultados[instancia]['pals']
        
        print "%d,%.1f,%.1f,%.1f,%.1f,%.1f" % (str(instancia), \
            minmin_time, pminmin_time, pals_time, \
            100-(pminmin_time * 100 / minmin_time), \
            100-(pals_time * 100 / minmin_time))
