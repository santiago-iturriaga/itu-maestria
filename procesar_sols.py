#encoding: utf-8

import sys
import os
import math

cant_iters=30
list_heur_dir = 'list-heuristics/'
pals_ruso_dir = 'pals-ruso/'

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Error! Uso: %s <dimension> <sufijo>"
        print "        Ej: %s 1024x32 4"
        exit(-1)

    dimension = sys.argv[1]
    sufijo = sys.argv[2]

    list_heur_dir = list_heur_dir + dimension
    pals_ruso_dir = pals_ruso_dir + dimension
    pals_dir = dimension + '.' + sufijo

    print 'List heuristics path : %s' % list_heur_dir
    print 'PALS ruso path       : %s' % pals_ruso_dir
    print 'PALS path            : %s' % pals_dir

    instancias_raw = []

    for filename in os.listdir(pals_dir):
        nameParts = filename.split('.')
        instancias_raw.append((nameParts[1],nameParts[3]+'.'+nameParts[4]))

    instancias = list(set(instancias_raw))
    instancias.sort()

    resultados_MinMin = {}
    resultados_MINMin = {}
    resultados_MinMIN = {}
    resultados_MINMIN = {}
    resultados_pals = {}
    resultados_pals_info = {}
    resultados_pals_ruso = {}

    for instancia in instancias:
        path = list_heur_dir + '/MinMin.scenario.' + instancia[0] + '.workload.' + instancia[1] + '.metrics'
        #print path

        if os.path.isfile(path):
            metrics_file = open(path)
            values = metrics_file.readline().split(' ')
            makespan = float(values[0])
            energy = float(values[1])

            resultados_MinMin[instancia] = (makespan, energy)
        else:
            print "[ERROR] cargando heuristica MinMin"
            exit(-1)

    for instancia in instancias:
        path = list_heur_dir + '/MinMIN.scenario.' + instancia[0] + '.workload.' + instancia[1] + '.metrics'
        #print path

        if os.path.isfile(path):
            metrics_file = open(path)
            values = metrics_file.readline().split(' ')
            #print values
            makespan = float(values[0])
            energy = float(values[1])

            resultados_MinMIN[instancia] = (makespan, energy)
        else:
            print "[ERROR] cargando heuristica MinMIN"
            exit(-1)

    for instancia in instancias:
        path = list_heur_dir + '/MINMin.scenario.' + instancia[0] + '.workload.' + instancia[1] + '.metrics'
        #print path

        if os.path.isfile(path):
            metrics_file = open(path)
            values = metrics_file.readline().split(' ')
            makespan = float(values[0])
            energy = float(values[1])

            resultados_MINMin[instancia] = (makespan, energy)
        else:
            print "[ERROR] cargando heuristica MINMin"
            exit(-1)

    for instancia in instancias:
        path = list_heur_dir + '/MINMIN.scenario.' + instancia[0] + '.workload.' + instancia[1] + '.metrics'
        #print path

        if os.path.isfile(path):
            metrics_file = open(path)
            values = metrics_file.readline().split(' ')
            makespan = float(values[0])
            energy = float(values[1])

            resultados_MINMIN[instancia] = (makespan, energy)
        else:
            print "[ERROR] cargando heuristica MINMIN"
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

        total_makespan = 0.0
        total_energy = 0.0
        total_sols = 0

        aux_iter_metrics = []

        total_time = 0.0

        for iter in range(cant_iters):
            dir_path = pals_dir + '/scenario.' + instancia[0] + '.workload.' + instancia[1] + '.' + str(iter) + '/'
            file_name = 'pals-1.scenario.' + instancia[0] + '.workload.' + instancia[1] + '.metrics'

            path = dir_path + file_name
            #print path

            if os.path.isfile(path):
                metrics_file = open(path)

                sols = 0
                min_makespan = 0.0
                min_energy = 0.0

                for line in metrics_file:
                    values = line.split(' ')
                    makespan = float(values[0])
                    energy = float(values[1])
                    sols = sols + 1

                    if min_makespan == 0.0: min_makespan = makespan
                    elif min_makespan > makespan: min_makespan = makespan

                    if min_energy == 0.0: min_energy = energy
                    elif min_energy > energy: min_energy = energy

                if abs_min_makespan == 0.0: abs_min_makespan = min_makespan
                elif abs_min_makespan > min_makespan: abs_min_makespan = min_makespan

                if abs_min_energy == 0.0: abs_min_energy = min_energy
                elif abs_min_energy > min_energy: abs_min_energy = min_energy

                aux_iter_metrics.append((sols, min_makespan, min_energy))

                total_makespan = total_makespan + min_makespan
                total_energy = total_energy + min_energy
                total_sols = total_sols + sols
            else:
                print "[ERROR] cargando heuristica pals"
                #exit(-1)

            avg_makespan = total_makespan/cant_iters
            avg_energy = total_energy/cant_iters

            aux_stdev_makespan = 0.0
            aux_stdev_energy = 0.0

            for (cant_s, min_m, min_e) in aux_iter_metrics:
                aux_stdev_makespan = aux_stdev_makespan + math.pow(min_m - avg_makespan, 2)
                aux_stdev_energy = aux_stdev_energy + math.pow(min_e - avg_energy, 2)

            stdev_makespan = math.sqrt((1.0/(cant_iters-1.0))*aux_stdev_makespan)
            stdev_energy = math.sqrt((1.0/(cant_iters-1.0))*aux_stdev_energy)

            resultados_pals[instancia] = (abs_min_makespan, abs_min_energy, total_sols/cant_iters, avg_makespan, stdev_makespan, avg_energy, stdev_energy)

            # INFO ===========================

            file_name = 'pals-1.scenario.' + instancia[0] + '.workload.' + instancia[1] + '.info'

            path = dir_path + file_name
            #print path

            if os.path.isfile(path):
                info_file = open(path)

                for line in info_file:
                    values = line.strip().split('|')

                    if values[0] == 'TOTAL_TIME':
                        total_time = total_time + (float(values[1]) / 1000000.0)

            else:
                print "[ERROR] cargando info de la heuristica pals"
                #exit(-1)

        resultados_pals_info[instancia] = (total_time/cant_iters,)

    print "[====== Tabla de makespan ======]"
    print "Instancia,MinMin,MinMIN,MINMin,MINMIN,PALS Ruso,PALS Ruso vs MinMin,PALS 2obj,PALS 2obj vs MinMin, Avg PALS 2obj, Stdev PALS 2obj, Avg ND"
    for instancia in instancias:
        min_minmin = resultados_MinMin[instancia][0]
        if resultados_MinMIN[instancia][0] < min_minmin: min_minmin = resultados_MinMIN[instancia][0]
        if resultados_MINMin[instancia][0] < min_minmin: min_minmin = resultados_MINMin[instancia][0]
        if resultados_MINMIN[instancia][0] < min_minmin: min_minmin = resultados_MINMIN[instancia][0]

        print "%s,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%s" % ('s' + instancia[0] + ' ' + instancia[1], \
            resultados_MinMin[instancia][0], \
            resultados_MinMIN[instancia][0], \
            resultados_MINMin[instancia][0], \
            resultados_MINMIN[instancia][0], \
            resultados_pals_ruso[instancia][0], \
            100.0 - (resultados_pals_ruso[instancia][0] * 100.0 / min_minmin), \
            resultados_pals[instancia][0], \
            100.0 - (resultados_pals[instancia][0] * 100.0 / min_minmin), \
            resultados_pals[instancia][3], \
            resultados_pals[instancia][4] * 100.0 / resultados_pals[instancia][3], \
            resultados_pals[instancia][2])

    print "[====== Tabla de energía ======]"
    print "Instancia,MinMin,MinMIN,MINMin,MINMIN,PALS Ruso,PALS Ruso vs MinMin,PALS 2obj,PALS 2obj vs MinMin"
    for instancia in instancias:
        min_minmin = resultados_MinMin[instancia][1]
        if resultados_MinMIN[instancia][1] < min_minmin: min_minmin = resultados_MinMIN[instancia][1]
        if resultados_MINMin[instancia][1] < min_minmin: min_minmin = resultados_MINMin[instancia][1]
        if resultados_MINMIN[instancia][1] < min_minmin: min_minmin = resultados_MINMIN[instancia][1]

        print "%s,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%s" % ('s' + instancia[0] + ' ' + instancia[1], \
            resultados_MinMin[instancia][1], \
            resultados_MinMIN[instancia][1], \
            resultados_MINMin[instancia][1], \
            resultados_MINMIN[instancia][1], \
            resultados_pals_ruso[instancia][1], \
            100.0 - (resultados_pals_ruso[instancia][1] * 100.0 / min_minmin), \
            resultados_pals[instancia][1], \
            100.0 - (resultados_pals[instancia][1] * 100.0 / min_minmin), \
            resultados_pals[instancia][5], \
            resultados_pals[instancia][6] * 100.0 / resultados_pals[instancia][5], \
            resultados_pals[instancia][2])

    print "[====== Tabla de info ======]"
    print "Instancia,Avg time"
    for instancia in instancias:
        print "%s,%.1f" % ('s' + instancia[0] + ' ' + instancia[1], resultados_pals_info[instancia][0])

    instancias_grupo_1 = {}
    instancias_grupo_2 = {}
    instancias_grupo_3 = {}

    for instancia in instancias:
        workload_parts = instancia[1].split('.')
        workload_model = workload_parts[0]
        workload_type = workload_parts[1].split('_')

        grupo_1 = (workload_model, workload_type[1])
        grupo_2 = (workload_model, workload_type[2])
        grupo_3 = (workload_type[1], workload_type[2])

        if not grupo_1 in instancias_grupo_1: instancias_grupo_1[grupo_1] = []
        if not grupo_2 in instancias_grupo_2: instancias_grupo_2[grupo_2] = []
        if not grupo_3 in instancias_grupo_3: instancias_grupo_3[grupo_3] = []

        instancias_grupo_1[grupo_1].append(instancia)
        instancias_grupo_2[grupo_2].append(instancia)
        instancias_grupo_3[grupo_3].append(instancia)

    print "[====== Tabla GRUPO 1 ======]"
    for item_grupo in sorted(instancias_grupo_1.keys()):
        items = float(len(instancias_grupo_1[item_grupo]))

        mk_total_improvement_best = 0.0
        mk_total_improvement_avg = 0.0
        mk_total_std_dev = 0.0
        mk_total_nd = 0

        nrg_total_improvement_best = 0.0
        nrg_total_improvement_avg = 0.0
        nrg_total_std_dev = 0.0
        nrg_total_nd = 0

        for instancia in instancias_grupo_1[item_grupo]:
            min_minmin = resultados_MinMin[instancia][0]
            if resultados_MinMIN[instancia][0] < min_minmin: min_minmin = resultados_MinMIN[instancia][0]
            if resultados_MINMin[instancia][0] < min_minmin: min_minmin = resultados_MINMin[instancia][0]
            if resultados_MINMIN[instancia][0] < min_minmin: min_minmin = resultados_MINMIN[instancia][0]

            mk_total_improvement_best = mk_total_improvement_best + (100.0 - (resultados_pals[instancia][0] * 100.0 / min_minmin))
            mk_total_improvement_avg = mk_total_improvement_avg + (100.0 - (resultados_pals[instancia][3] * 100.0 / min_minmin))
            mk_total_std_dev = mk_total_std_dev + (resultados_pals[instancia][4] * 100.0 / resultados_pals[instancia][3])
            mk_total_nd = mk_total_nd + resultados_pals[instancia][2]

            min_minmin = resultados_MinMin[instancia][1]
            if resultados_MinMIN[instancia][1] < min_minmin: min_minmin = resultados_MinMIN[instancia][1]
            if resultados_MINMin[instancia][1] < min_minmin: min_minmin = resultados_MINMin[instancia][1]
            if resultados_MINMIN[instancia][1] < min_minmin: min_minmin = resultados_MINMIN[instancia][1]

            nrg_total_improvement_best = nrg_total_improvement_best + (100.0 - (resultados_pals[instancia][1] * 100.0 / min_minmin))
            nrg_total_improvement_avg = nrg_total_improvement_avg + (100.0 - (resultados_pals[instancia][5] * 100.0 / min_minmin))
            nrg_total_std_dev = nrg_total_std_dev + (resultados_pals[instancia][6] * 100.0 / resultados_pals[instancia][5])
            nrg_total_nd = nrg_total_nd + resultados_pals[instancia][2]

        #print "%s,%s,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f" % (item_grupo[0], item_grupo[1], \
        #    mk_total_improvement_best / items, \
        #    nrg_total_improvement_best / items, \
        #    mk_total_improvement_avg / items, \
        #    nrg_total_improvement_avg / items, \
        #    mk_total_std_dev / items, \
        #    nrg_total_std_dev / items, \
        #    mk_total_nd / items)

        model_desc = ""
        type_desc = ""
        if item_grupo[0] == 'A': model_desc = 'Ali \emph{et al.}'
        if item_grupo[0] == 'B': model_desc = 'Braun \emph{et al.}'
        if item_grupo[1] == 'c': type_desc = 'consistent'
        if item_grupo[1] == 's': type_desc = 'semiconsistent'
        if item_grupo[1] == 'i': type_desc = 'inconsistent'
        print "%s & %s & %.1f \\%% & %.1f \\%% \\\\" % (model_desc, type_desc, mk_total_improvement_avg / items, nrg_total_improvement_avg / items)

    print "[====== Tabla GRUPO 2 ======]"
    for item_grupo in sorted(instancias_grupo_2.keys()):
        items = float(len(instancias_grupo_2[item_grupo]))

        mk_total_improvement_best = 0.0
        mk_total_improvement_avg = 0.0
        mk_total_std_dev = 0.0
        mk_total_nd = 0

        nrg_total_improvement_best = 0.0
        nrg_total_improvement_avg = 0.0
        nrg_total_std_dev = 0.0
        nrg_total_nd = 0

        for instancia in instancias_grupo_2[item_grupo]:
            min_minmin = resultados_MinMin[instancia][0]
            if resultados_MinMIN[instancia][0] < min_minmin: min_minmin = resultados_MinMIN[instancia][0]
            if resultados_MINMin[instancia][0] < min_minmin: min_minmin = resultados_MINMin[instancia][0]
            if resultados_MINMIN[instancia][0] < min_minmin: min_minmin = resultados_MINMIN[instancia][0]

            mk_total_improvement_best = mk_total_improvement_best + (100.0 - (resultados_pals[instancia][0] * 100.0 / min_minmin))
            mk_total_improvement_avg = mk_total_improvement_avg + (100.0 - (resultados_pals[instancia][3] * 100.0 / min_minmin))
            mk_total_std_dev = mk_total_std_dev + (resultados_pals[instancia][4] * 100.0 / resultados_pals[instancia][3])
            mk_total_nd = mk_total_nd + resultados_pals[instancia][2]

            min_minmin = resultados_MinMin[instancia][1]
            if resultados_MinMIN[instancia][1] < min_minmin: min_minmin = resultados_MinMIN[instancia][1]
            if resultados_MINMin[instancia][1] < min_minmin: min_minmin = resultados_MINMin[instancia][1]
            if resultados_MINMIN[instancia][1] < min_minmin: min_minmin = resultados_MINMIN[instancia][1]

            nrg_total_improvement_best = nrg_total_improvement_best + (100.0 - (resultados_pals[instancia][1] * 100.0 / min_minmin))
            nrg_total_improvement_avg = nrg_total_improvement_avg + (100.0 - (resultados_pals[instancia][5] * 100.0 / min_minmin))
            nrg_total_std_dev = nrg_total_std_dev + (resultados_pals[instancia][6] * 100.0 / resultados_pals[instancia][5])
            nrg_total_nd = nrg_total_nd + resultados_pals[instancia][2]

        #print "%s,%s,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f" % (item_grupo[0], item_grupo[1], \
        #    mk_total_improvement_best / items, \
        #    nrg_total_improvement_best / items, \
        #    mk_total_improvement_avg / items, \
        #    nrg_total_improvement_avg / items, \
        #    mk_total_std_dev / items, \
        #    nrg_total_std_dev / items, \
        #    mk_total_nd / items)
            
        model_desc = ""
        type_desc = ""
        if item_grupo[0] == 'A': model_desc = 'Ali \emph{et al.}'
        if item_grupo[0] == 'B': model_desc = 'Braun \emph{et al.}'
        if item_grupo[1] == 'hihi': type_desc = 'High High'
        if item_grupo[1] == 'hilo': type_desc = 'High Low'
        if item_grupo[1] == 'lohi': type_desc = 'Low High'
        if item_grupo[1] == 'lolo': type_desc = 'Low Low'
        print "%s & %s & %.1f \\%% & %.1f \\%% \\\\" % (model_desc, type_desc, mk_total_improvement_avg / items, nrg_total_improvement_avg / items)

    csv = ""

    print "[====== Tabla GRUPO 3 ======]"
    for item_grupo in sorted(instancias_grupo_3.keys()):
        items = float(len(instancias_grupo_3[item_grupo]))

        mk_total_improvement_best = 0.0
        mk_total_improvement_avg = 0.0
        mk_total_std_dev = 0.0
        mk_total_nd = 0

        nrg_total_improvement_best = 0.0
        nrg_total_improvement_avg = 0.0
        nrg_total_std_dev = 0.0
        nrg_total_nd = 0

        for instancia in instancias_grupo_3[item_grupo]:
            min_minmin = resultados_MinMin[instancia][0]
            if resultados_MinMIN[instancia][0] < min_minmin: min_minmin = resultados_MinMIN[instancia][0]
            if resultados_MINMin[instancia][0] < min_minmin: min_minmin = resultados_MINMin[instancia][0]
            if resultados_MINMIN[instancia][0] < min_minmin: min_minmin = resultados_MINMIN[instancia][0]

            mk_total_improvement_best = mk_total_improvement_best + (100.0 - (resultados_pals[instancia][0] * 100.0 / min_minmin))
            mk_total_improvement_avg = mk_total_improvement_avg + (100.0 - (resultados_pals[instancia][3] * 100.0 / min_minmin))
            mk_total_std_dev = mk_total_std_dev + (resultados_pals[instancia][4] * 100.0 / resultados_pals[instancia][3])
            mk_total_nd = mk_total_nd + resultados_pals[instancia][2]

            min_minmin = resultados_MinMin[instancia][1]
            if resultados_MinMIN[instancia][1] < min_minmin: min_minmin = resultados_MinMIN[instancia][1]
            if resultados_MINMin[instancia][1] < min_minmin: min_minmin = resultados_MINMin[instancia][1]
            if resultados_MINMIN[instancia][1] < min_minmin: min_minmin = resultados_MINMIN[instancia][1]

            nrg_total_improvement_best = nrg_total_improvement_best + (100.0 - (resultados_pals[instancia][1] * 100.0 / min_minmin))
            nrg_total_improvement_avg = nrg_total_improvement_avg + (100.0 - (resultados_pals[instancia][5] * 100.0 / min_minmin))
            nrg_total_std_dev = nrg_total_std_dev + (resultados_pals[instancia][6] * 100.0 / resultados_pals[instancia][5])
            nrg_total_nd = nrg_total_nd + resultados_pals[instancia][2]

        csv = csv + "%s,%s,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n" % (item_grupo[0], item_grupo[1], \
            mk_total_improvement_best / items, \
            nrg_total_improvement_best / items, \
            mk_total_improvement_avg / items, \
            nrg_total_improvement_avg / items, \
            mk_total_std_dev / items, \
            nrg_total_std_dev / items, \
            mk_total_nd / items)
            
        model_desc = ""
        type_desc = ""
        if item_grupo[0] == 'c': model_desc = 'consistent'
        if item_grupo[0] == 's': model_desc = 'semiconsistent'
        if item_grupo[0] == 'i': model_desc = 'inconsistent'
        if item_grupo[1] == 'hihi': type_desc = 'High High'
        if item_grupo[1] == 'hilo': type_desc = 'High Low'
        if item_grupo[1] == 'lohi': type_desc = 'Low High'
        if item_grupo[1] == 'lolo': type_desc = 'Low Low'
        print "%s & %s & $%.1f \\%%$ & $%.1f \\%%$ \\\\" % (model_desc, type_desc, mk_total_improvement_avg / items, nrg_total_improvement_avg / items)

    print "[====== Tabla improvements makespan ======]"
    print "Instancia,MinMin,MinMIN,MINMin,MINMIN"
    avg_makespan = [0.0,0.0,0.0,0.0]
    for instancia in instancias:
        avg_makespan[0] = avg_makespan[0] + (100 - (resultados_pals[instancia][3] * 100 / resultados_MinMin[instancia][0]))
        avg_makespan[1] = avg_makespan[1] + (100 - (resultados_pals[instancia][3] * 100 / resultados_MinMIN[instancia][0]))
        avg_makespan[2] = avg_makespan[2] + (100 - (resultados_pals[instancia][3] * 100 / resultados_MINMin[instancia][0]))
        avg_makespan[3] = avg_makespan[3] + (100 - (resultados_pals[instancia][3] * 100 / resultados_MINMIN[instancia][0]))
        
    print "%s,%.1f,%.1f,%.1f,%.1f" % ('s' + instancia[0] + ' ' + instancia[1], \
        avg_makespan[0]/len(instancias), avg_makespan[1]/len(instancias), avg_makespan[2]/len(instancias), avg_makespan[3]/len(instancias))

    print "[====== Tabla de energía ======]"
    print "Instancia,MinMin,MinMIN,MINMin,MINMIN"
    avg_energy = [0.0,0.0,0.0,0.0]
    for instancia in instancias:
        avg_energy[0] = avg_energy[0] + (100 - (resultados_pals[instancia][5] * 100 / resultados_MinMin[instancia][1]))
        avg_energy[1] = avg_energy[1] + (100 - (resultados_pals[instancia][5] * 100 / resultados_MinMIN[instancia][1]))
        avg_energy[2] = avg_energy[2] + (100 - (resultados_pals[instancia][5] * 100 / resultados_MINMin[instancia][1]))
        avg_energy[3] = avg_energy[3] + (100 - (resultados_pals[instancia][5] * 100 / resultados_MINMIN[instancia][1]))

    print "%s,%.1f,%.1f,%.1f,%.1f" % ('s' + instancia[0] + ' ' + instancia[1], \
        avg_energy[0]/len(instancias), avg_energy[1]/len(instancias), avg_energy[2]/len(instancias), avg_energy[3]/len(instancias))

    print csv
