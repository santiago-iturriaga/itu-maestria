#encoding: utf-8

import os
import math

cantidad_instancias = 20
algoritmos = ['pals+gminmin'] #,'minmin', 'pals+mct', 'pals+pminmin+12']

if __name__ == '__main__':
    resultados = {}

    for instancia in range(21)[1:]:
        resultados[instancia] = {}

        for a in algoritmos:
            base_path = 'solutions.by_time/' + str(instancia) + '.' + a
            print base_path

            if os.path.isfile(base_path + '.makespan'):
                dmake_file = open(base_path + '.makespan')
                dmake = float(dmake_file.readline())

                dtime_file = open(base_path + '.time')
                dtime_lines = dtime_file.readlines()

                milestones = []

                host_init_time = 0.0
                sol_init_type = ''
                sol_init_time = 0.0
                device_init_time = 0.0

                for line in dtime_lines:
                    if line.split('\t')[0].strip() == 'real':
                        dtime_str = line.split('\t')[1].strip()
                        dtime_mins = int(dtime_str.split('m')[0].strip())
                        dtime_secs = float(dtime_str.split('m')[1].strip().strip('s').strip())
                        dtime = dtime_mins * 60 + dtime_secs

                    if line.split('|')[0].strip() == 'LOADING(s)':
                        host_init_time = float(line.split('|')[1].strip())

                    if line.split('|')[0].strip() == 'INIT':
                        sol_init_type = line.split('|')[1].strip()

                    if line.split('|')[0].strip() == 'INIT(s)':
                        sol_init_time = float(line.split('|')[1].strip())

                    if line.split('|')[0].strip() == 'GPU_LOADING(s)':
                        device_init_time = float(line.split('|')[1].strip())

                    if line.split('|')[0].strip() == 'MAKESPAN':
                        milestone_time = int(line.split('|')[1].strip())
                        milestone_iteration = int(line.split('|')[2].strip())
                        milestone_value = float(line.split('|')[3].strip())

                        milestones.append((milestone_time, milestone_iteration, milestone_value))

                resultados[instancia][a] = (dmake, dtime, host_init_time, \
                    sol_init_type, sol_init_time, device_init_time, milestones)
            else:
                exit(-1)

    print resultados

    print "====== Tabla de makespan ======"
    print "Instancia,MinMin,PALS MCT,PALS pMinMin,Improv. PALS MCT vs MinMin,Improv. PALS pMinMin vs MinMin"
    for instancia in range(21)[1:]:
        #mct_make = resultados[instancia]['mct'][0]
        minmin_make = resultados[instancia]['minmin'][0]
        #pminmin_make = resultados[instancia]['pminmin'][0]
        pals_mct_make = resultados[instancia]['pals+mct'][0]
        pals_pminmin_make = resultados[instancia]['pals+pminmin+12'][0]

        print "%d,%.1f,%.1f,%.1f,%.1f,%.1f" % (instancia, \
            minmin_make, pals_mct_make, pals_pminmin_make, \
            100-(pals_mct_make * 100 / minmin_make), \
            100-(pals_pminmin_make * 100 / minmin_make))

    print "====== Tabla de tiempos (segundos) ======"
    print "Instancia,MinMin,PALS MCT,PALS pMinMin,Speedup PALS MCT,Speedup PALS pMinMin"
    for instancia in range(21)[1:]:
        minmin_time = resultados[instancia]['minmin'][1]
        pals_mct_time = resultados[instancia]['pals+mct'][1]
        pals_pminmin_time = resultados[instancia]['pals+pminmin+12'][1]

        print "%d,%.1f,%.1f,%.1f,%.1f,%.1f" % (instancia, \
            minmin_time, pals_pminmin_time, pals_mct_time, \
            minmin_time / pals_pminmin_time, \
            minmin_time / pals_mct_time)

    print "====== Ejemplo evolución ======"
    print "Iteracion,PALS MCT,PALS pMinMin"
    pals_mct_miles = resultados[1]['pals+mct'][6]
    pals_pminmin_miles = resultados[1]['pals+pminmin+12'][6]

    tope = len(pals_mct_miles)
    if len(pals_pminmin_miles) < tope: tope = len(pals_pminmin_miles)

    for index in range(tope):
        (mct_milestone_time, mct_milestone_iteration, mct_milestone_value) = pals_mct_miles[index]
        (pminmin_milestone_time, pminmin_milestone_iteration, pminmin_milestone_value) = pals_pminmin_miles[index]
        
        print "%d,%.1f,%.1f" % (mct_milestone_iteration, \
            mct_milestone_value, pminmin_milestone_value)
