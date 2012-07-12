#encoding: utf-8

import os
import math

cantidad_instancias = 20
algoritmos = ['pals+mct', 'pals+pminmin+12']

if __name__ == '__main__':
    resultados = {}

    for instancia in range(21)[1:]:
        resultados[instancia] = {}

        for a in algoritmos:
            base_path = 'solutions.by_iters/' + str(instancia) + '.' + a
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

    print "====== Ejemplo evolución ======"
    print "Tiempo(s),PALS MCT,PALS pMinMin"
    
    valores_mct = []
    valores_pminmin = []

    pals_mct_starting = 0
    pals_pminmin_starting = 0

    (mct_milestone_time, mct_milestone_iteration, mct_milestone_value) = pals_mct_miles[1]
    (pminmin_milestone_time, pminmin_milestone_iteration, pminmin_milestone_value) = pals_pminmin_miles[1]
      
    pals_mct_starting = mct_milestone_time
    pals_pminmin_starting = pminmin_milestone_time

    valores_mct.append(mct_milestone_value)
    valores_pminmin.append(pminmin_milestone_value)

    for index in range(len(pals_mct_miles))[2:]:
	(mct_milestone_time, mct_milestone_iteration, mct_milestone_value) = pals_mct_miles[index]
      
	if mct_milestone_time - pals_mct_starting >= len(valores_mct):
	    valores_mct.append(mct_milestone_value)
	    
    for index in range(len(pals_pminmin_miles))[2:]:
        (pminmin_milestone_time, pminmin_milestone_iteration, pminmin_milestone_value) = pals_pminmin_miles[index]
      
	if pminmin_milestone_time - pals_pminmin_starting >= len(valores_pminmin):
	    valores_pminmin.append(pminmin_milestone_value)

    tope = len(valores_mct)
    if len(valores_pminmin) < tope: tope = len(valores_pminmin)

    for index in range(tope):
        print "%d,%.1f,%.1f" % (index, \
            valores_mct[index], valores_pminmin[index])
	    

