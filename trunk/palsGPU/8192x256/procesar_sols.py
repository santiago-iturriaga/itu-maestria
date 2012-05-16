#encoding: utf-8

import os
import math

cantidad_instancias = 20
algoritmos = ['pminmin', 'minmin', 'pals']

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

    stddev = 0 #math.sqrt(aux_valor / (len(valores)-1))

    return (min, max, avg, stddev)

  if __name__ == '__main__':
    resultados = []

    for instancia in range(21)[1:]:
        for a in algoritmo:
            base_path = 'solutions/' + instancia + '.' + a
            print base_path

            if os.path.isfile(base_path + '.makespan'):
                dmake_file = open(base_path + '.makespan')
                dmake = float(dmake_file.readline())

                dtime_file = open(base_path + '.time')
                dtime_lines = dtime_file.readlines()
                dtime_line = dtime_lines[1].strip()
                dtime_str = dtime_line.split('\t')[1].strip()
                dtime_mins = int(dtime_str.split('m')[0].strip())
                dtime_secs = float(dtime_str.split('m')[1].strip().strip('s').strip())
                dtime = dtime_mins * 60 + dtime_secs
		if a == 'pals':
  dtime_line = dtime_lines[1].strip()
				                  dtime_str = dtime_line.split('\t')[1].strip()
						                  dtime_mins = int(dtime_str.split('m')[0].strip())
								                  dtime_secs = float(dtime_str.split('m')[1].strip().strip('s').strip())
										                  dtime = dtime_mins * 60 + dtime_secs

		else:


                resultados_deterministas.append((instancia, d.strip('.'), dmake, dtime))
            else:
                exit(-1)

        for iter in range(ITER_GPU):
            base_path = 'solutions/' + instancia + '.palsGPU.' + str(iter)
            #print base_path

            if os.path.isfile(base_path  + '.makespan'):
                dmake_file = open(base_path  + '.makespan')
                dmake = float(dmake_file.readline())

                dtime_file = open(base_path  + '.time')
                dtime_lines = dtime_file.readlines()
                dtime_line = dtime_lines[1].strip()
                dtime_str = dtime_line.split('\t')[1].strip()
                dtime_mins = int(dtime_str.split('m')[0].strip())
                dtime_secs = float(dtime_str.split('m')[1].strip().strip('s').strip())
                dtime = dtime_mins * 60 + dtime_secs

                dinfo_file = open(base_path  + '.info')
                dinfo_cant_iter = int(dinfo_file.readline().split('|')[1].strip())
                dinfo_best_found = int(dinfo_file.readline().split('|')[1].strip())

                resultados_palsGPU.append((instancia, dmake, dtime, dinfo_cant_iter, dinfo_best_found))
            else:
                exit(-1)

    #print resultados_deterministas
    #print resultados_palsGPU

    medidas_palsGPU = {}
    medidas_deterministas = {}

    for instancia in instancias:
        make_values = []
        time_values = []
        cant_iter_values = []
	best_found_values = []

        for (r_inst, r_make, r_time, r_cant_iter, r_best_found) in resultados_palsGPU:
            if r_inst == instancia:
                make_values.append(r_make)
                time_values.append(r_time)
                cant_iter_values.append(r_cant_iter)
                best_found_values.append(r_best_found)

        (make_min, make_max, make_avg, make_stddev) = calcular_medidas(make_values)
        (time_min, time_max, time_avg, time_stddev) = calcular_medidas(time_values)
        (cant_iter_min, cant_iter_max, cant_iter_avg, cant_iter_stddev) = calcular_medidas(cant_iter_values)
        (best_found_min, best_found_max, best_found_avg, best_found_stddev) = calcular_medidas(best_found_values)

        medidas_palsGPU[instancia] = [(make_min, make_max, make_avg, make_stddev), \
            (time_min, time_max, time_avg, time_stddev), \
            (cant_iter_min, cant_iter_max, cant_iter_avg, cant_iter_stddev),
            (best_found_min, best_found_max, best_found_avg, best_found_stddev)]

        medidas_deterministas[instancia] = {}

        for (r_inst, r_d, r_make, r_time) in resultados_deterministas:
            if r_inst == instancia:
                medidas_deterministas[instancia][r_d] = (r_make, r_time)

    print "====== Tabla de makespan ======"
    print "Instancia,MCT,Min-Min,Best PGPU,Avg PGPU,Stddev PGPU,Worst PGPU,Avg PGPU vs Min-Min"
    for instancia in instancias:
        (minmin_make, minmin_time) = medidas_deterministas[instancia]['minmin']
        (mct_make, mct_time) = medidas_deterministas[instancia]['mct']
        
        [(make_min, make_max, make_avg, make_stddev), \
        (time_min, time_max, time_avg, time_stddev), \
        (cant_iter_min, cant_iter_max, cant_iter_avg, cant_iter_stddev), \
        (best_found_min, best_found_max, best_found_avg, best_found_stddev)] = medidas_palsGPU[instancia]
        print "%s,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f" % (instancia, mct_make, minmin_make, make_min, make_avg, (make_stddev * 100 / make_avg), make_max, 100-(make_avg * 100 / minmin_make))

    print "====== Tabla de tiempos (segundos) ======"
    print "Instancia,MCT,Min-Min,Min time PGPU,Avg time PGPU,Stddev time PGPU,Worst time PGPU,Avg time PGPU vs Min-Min"
    total_avg_time = 0.0
    for instancia in instancias:
        (minmin_make, minmin_time) = medidas_deterministas[instancia]['minmin']
        (mct_make, mct_time) = medidas_deterministas[instancia]['mct']
        
        [(make_min, make_max, make_avg, make_stddev), \
        (time_min, time_max, time_avg, time_stddev), \
        (cant_iter_min, cant_iter_max, cant_iter_avg, cant_iter_stddev), \
        (best_found_min, best_found_max, best_found_avg, best_found_stddev)] = medidas_palsGPU[instancia]
        print "%s,%.4f,%.4f,%.4f,%.4f,%.1f,%.4f,%.1f" % (instancia, mct_time, minmin_time, time_min, time_avg, (time_stddev * 100 / time_avg), time_max, 100-(time_avg * 100 / minmin_time))

        total_avg_time = total_avg_time + time_avg

    print ">> Total avg time: %.4f" % total_avg_time

    print "====== Tabla de iteraciones ======"
    print "Instancia,Min iter PGPU,Avg iter PGPU,Stddev iter PGPU,Worst iter PGPU,Best found min, Best found avg"
    for instancia in instancias:       
        [(make_min, make_max, make_avg, make_stddev), \
        (time_min, time_max, time_avg, time_stddev), \
        (cant_iter_min, cant_iter_max, cant_iter_avg, cant_iter_stddev), \
        (best_found_min, best_found_max, best_found_avg, best_found_stddev)] = medidas_palsGPU[instancia]
        print "%s,%d,%.1f,%.1f,%d,%d,%.1f" % (instancia, cant_iter_min, cant_iter_avg, (cant_iter_stddev * 100 / cant_iter_avg), cant_iter_max, best_found_min, best_found_avg)
