#encoding: utf-8

import sys
import os
import math

cant_iters=15

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print "Error! Uso: %s <dimension> <sufijo> <start> <end> <prefijo>"
        print "        Ej: %s 1024x32 speed 1 24"
        exit(-1)

    dimension = sys.argv[1]
    sufijo = sys.argv[2]
    start = int(sys.argv[3])
    end = int(sys.argv[4])
    prefix = sys.argv[5]

    avg_thread_0 = 0.0

    for threads in range(start,end+1):
        pals_dir = dimension + '.' + sufijo + '.' + str(threads)
        #print 'PALS path            : %s' % pals_dir

        instancias_raw = []

        for filename in os.listdir(pals_dir):
            nameParts = filename.split('.')
            instancias_raw.append((nameParts[1],nameParts[3]+'.'+nameParts[4]))

        instancias = list(set(instancias_raw))
        instancias.sort()

        resultados_pals_info = {}

        for instancia in instancias:
            total_time = 0.0
            
            valores = []

            for iter in range(cant_iters):
                dir_path = pals_dir + '/scenario.' + instancia[0] + '.workload.' + instancia[1] + '.' + str(iter) + '/'
                file_name = prefix + '.scenario.' + instancia[0] + '.workload.' + instancia[1] + '.metrics'

                path = dir_path + file_name
                
                #print path

                # INFO ===========================

                file_name = prefix + '.scenario.' + instancia[0] + '.workload.' + instancia[1] + '.info'

                path = dir_path + file_name
                #print path

                if os.path.isfile(path):
                    info_file = open(path)

                    for line in info_file:
                        values = line.strip().split('|')

                        if values[0] == 'TOTAL_TIME':
                            total_time = total_time + (float(values[1]) / 1000000.0)
                            valores.append(float(values[1]) / 1000000.0)

                else:
                    print "[ERROR] cargando info de la heuristica pals"
                    #exit(-1)
                    
            avg = total_time/cant_iters
            
            aux = 0.0
            for v in valores:
                aux = aux + math.pow(v-avg,2)
            aux = aux / (cant_iters-1)
            std = math.sqrt(aux)
            
            if threads == start:
                avg_thread_0 = avg
            
            resultados_pals_info[instancia] = (avg,std,avg_thread_0/avg)

        #print "[====== Tabla de info ======]"
        #print "Instancia,Avg time"
        for instancia in instancias:
            #print "%d,%s,%.1f,%.2f,%.1f" % (threads,'s' + instancia[0] + ' ' + instancia[1], resultados_pals_info[instancia][0], resultados_pals_info[instancia][1], resultados_pals_info[instancia][2])
            print "%d,%.1f,%.2f,%.1f" % (threads, resultados_pals_info[instancia][0], resultados_pals_info[instancia][1], resultados_pals_info[instancia][2])
