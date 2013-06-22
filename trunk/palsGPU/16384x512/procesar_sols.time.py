#encoding: utf-8

import os
import math

cantidad_instancias = 20
#algoritmos = ['pminmin', 'minmin', 'pals+mct', 'pals+pminmin+12']
#algoritmos = ['pals+mct', 'pals+pminmin+12']
algoritmos = ['pals+gminmin']

if __name__ == '__main__':
    resultados = {}

    for instancia in range(21)[1:]:
        resultados[instancia] = {}

        for a in algoritmos:
            dtime = 0.0
            dmake = 0.0
            dmake_list = []

            for t in range(30):
                base_path = 'solutions/' + str(instancia) + '.' + a + '.' + str(t)
                print base_path

                if os.path.isfile(base_path + '.makespan'):
                    dmake_file = open(base_path + '.makespan')
                    dmake_list.append(float(dmake_file.readline()))
                    dmake = dmake + dmake_list[t]

                    dtime_file = open(base_path + '.time')
                    dtime_lines = dtime_file.readlines()

                    #milestones = []

                    for line in dtime_lines:
                        if line.split('\t')[0].strip() == 'real':
                            dtime_str = line.split('\t')[1].strip()
                            dtime_mins = int(dtime_str.split('m')[0].strip())
                            dtime_secs = float(dtime_str.split('m')[1].strip().strip('s').strip())
                            dtime = dtime + dtime_mins * 60 + dtime_secs

                        #if line.split('|')[0].strip() == 'LOADING(s)':
                        #    host_init_time = float(line.split('|')[1].strip())

                        #if line.split('|')[0].strip() == 'INIT':
                        #    sol_init_type = line.split('|')[1].strip()

                        #if line.split('|')[0].strip() == 'INIT(s)':
                        #    sol_init_time = float(line.split('|')[1].strip())

                        #if line.split('|')[0].strip() == 'GPU_LOADING(s)':
                        #    device_init_time = float(line.split('|')[1].strip())

                        #if line.split('|')[0].strip() == 'MAKESPAN':
                        #    milestone_time = int(line.split('|')[1].strip())
                        #    milestone_iteration = int(line.split('|')[2].strip())
                        #    milestone_value = float(line.split('|')[3].strip())

                        #    milestones.append((milestone_time, milestone_iteration, milestone_value))

                else:
                    exit(-1)
                    
            sdiff = 0.0
            for value in dmake_list:
                sdiff = sdiff + pow(value - (dmake / 30.0), 2)
            
            resultados[instancia][a] = (dmake / 30.0, math.sqrt(sdiff / 29.0), dtime / 30.0)

    print resultados

    print "====== Tabla de makespan ======"
    print "Instancia,makecpan PALS gMinMin"
    for instancia in range(21)[1:]:
        pals_gminmin_makespan_avg = resultados[instancia]['pals+gminmin'][0]
        pals_gminmin_makespan_stdev = resultados[instancia]['pals+gminmin'][1]

        print "%d,%.1f,%.1f" % (instancia, pals_gminmin_makespan_avg, pals_gminmin_makespan_stdev)

    print "====== Tabla de tiempos (segundos) ======"
    print "Instancia,tiempo PALS gMinMin"
    for instancia in range(21)[1:]:
        #pals_mct_time = resultados[instancia]['pals+mct']
        #pals_pminmin_time = resultados[instancia]['pals+pminmin+12']
        pals_gminmin_time = resultados[instancia]['pals+gminmin'][2]

        #print "%d,%.1f,%.1f" % (instancia, pals_mct_time, pals_pminmin_time)
	print "%d,%.1f" % (instancia, pals_gminmin_time)

