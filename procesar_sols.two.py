#encoding: utf-8

import sys
import os
import math

cant_iters=30
list_heur_dir = 'list-heuristics/'

SCENARIOS=(0,3,6,9,10,11,13,14,16,17,19)
WORKLOADS=("A.u_c_hihi","A.u_c_hilo","A.u_c_lohi","A.u_c_lolo","A.u_i_hihi","A.u_i_hilo","A.u_i_lohi","A.u_i_lolo","A.u_s_hihi","A.u_s_hilo","A.u_s_lohi","A.u_s_lolo","B.u_c_hihi","B.u_c_hilo","B.u_c_lohi","B.u_c_lolo","B.u_i_hihi","B.u_i_hilo","B.u_i_lohi","B.u_i_lolo","B.u_s_hihi","B.u_s_hilo","B.u_s_lohi","B.u_s_lolo")

#SCENARIOS=(0,6,11,13,16,19)
#WORKLOADS=("A.u_c_hihi","A.u_c_hilo","A.u_c_lohi","A.u_c_lolo","B.u_c_hihi","B.u_c_hilo","B.u_c_lohi","B.u_c_lolo")

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print "Error! Uso: %s <dimension> <sufijo 1> <prefijo archivo 1> <sufijo 2> <prefijo archivo 2>"
        print "        Ej: %s 512x16 24_10s pals-aga 24.adhoc pals-1"
        exit(-1)

    dimension = sys.argv[1]
    
    sufijo_1 = sys.argv[2]
    prefijo_archivo_1 = sys.argv[3]
    
    sufijo_2 = sys.argv[4]
    prefijo_archivo_2 = sys.argv[5]

    list_heur_dir = list_heur_dir + dimension
   
    print 'List heuristics path : %s' % list_heur_dir

    instancias_raw = []

    for s in SCENARIOS:
        for w in WORKLOADS:
            instancias_raw.append((str(s),w))
            
    instancias = list(set(instancias_raw))
    instancias.sort()

    resultados_MinMin = {}
    resultados_MINMin = {}
    resultados_MinMIN = {}
    resultados_MINMIN = {}

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

    pals_dir = dimension + '.' + sufijo_1
    print 'PALS path            : %s' % pals_dir

    resultados_pals_1 = {}
    resultados_pals_info_1 = {}

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
            file_name = prefijo_archivo_1 + '.scenario.' + instancia[0] + '.workload.' + instancia[1] + '.metrics'

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
                print '[ERROR] cargando heuristica pals ' + path
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

            resultados_pals_1[instancia] = (abs_min_makespan, abs_min_energy, total_sols/cant_iters, avg_makespan, stdev_makespan, avg_energy, stdev_energy)

            # INFO ===========================

            file_name = prefijo_archivo_1 + '.scenario.' + instancia[0] + '.workload.' + instancia[1] + '.info'

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

        resultados_pals_info_1[instancia] = (total_time/cant_iters,)

    pals_dir = dimension + '.' + sufijo_2
    print 'PALS path            : %s' % pals_dir

    resultados_pals_2 = {}
    resultados_pals_info_2 = {}

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
            file_name = prefijo_archivo_2 + '.scenario.' + instancia[0] + '.workload.' + instancia[1] + '.metrics'

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
                print '[ERROR] cargando heuristica pals ' + path
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

            resultados_pals_2[instancia] = (abs_min_makespan, abs_min_energy, total_sols/cant_iters, avg_makespan, stdev_makespan, avg_energy, stdev_energy)

            # INFO ===========================

            file_name = prefijo_archivo_2 + '.scenario.' + instancia[0] + '.workload.' + instancia[1] + '.info'

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

        resultados_pals_info_2[instancia] = (total_time/cant_iters,)

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

    mk_best_1 = 0.0
    mk_avg_1 = 0.0
    nrg_best_1 = 0.0
    nrg_avg_1 = 0.0

    mk_best_2 = 0.0
    mk_avg_2 = 0.0
    nrg_best_2 = 0.0
    nrg_avg_2 = 0.0

    #print resultados_pals_1
    #print resultados_pals_2

    for item_grupo in sorted(instancias_grupo_2.keys()):
        items = float(len(instancias_grupo_2[item_grupo]))

        mk_total_improvement_best_1 = 0.0
        mk_total_improvement_avg_1 = 0.0
        #mk_total_std_dev = 0.0
        #mk_total_nd = 0

        nrg_total_improvement_best_1 = 0.0
        nrg_total_improvement_avg_1 = 0.0
        #nrg_total_std_dev = 0.0
        #nrg_total_nd = 0

        mk_total_improvement_best_2 = 0.0
        mk_total_improvement_avg_2 = 0.0

        nrg_total_improvement_best_2 = 0.0
        nrg_total_improvement_avg_2 = 0.0

        for instancia in instancias_grupo_2[item_grupo]:
            min_minmin = resultados_MinMin[instancia][0]
            if resultados_MinMIN[instancia][0] < min_minmin: min_minmin = resultados_MinMIN[instancia][0]
            if resultados_MINMin[instancia][0] < min_minmin: min_minmin = resultados_MINMin[instancia][0]
            if resultados_MINMIN[instancia][0] < min_minmin: min_minmin = resultados_MINMIN[instancia][0]

            mk_aux = 100.0 - (resultados_pals_1[instancia][0] * 100.0 / min_minmin)
            if mk_aux > mk_total_improvement_best_1:
                mk_total_improvement_best_1 = mk_aux
            mk_total_improvement_avg_1 = mk_total_improvement_avg_1 + (100.0 - (resultados_pals_1[instancia][3] * 100.0 / min_minmin))

            mk_aux = 100.0 - (resultados_pals_2[instancia][0] * 100.0 / min_minmin)
            if mk_aux > mk_total_improvement_best_2:
                mk_total_improvement_best_2 = mk_aux
            mk_total_improvement_avg_2 = mk_total_improvement_avg_2 + (100.0 - (resultados_pals_2[instancia][3] * 100.0 / min_minmin))

            min_minmin = resultados_MinMin[instancia][1]
            if resultados_MinMIN[instancia][1] < min_minmin: min_minmin = resultados_MinMIN[instancia][1]
            if resultados_MINMin[instancia][1] < min_minmin: min_minmin = resultados_MINMin[instancia][1]
            if resultados_MINMIN[instancia][1] < min_minmin: min_minmin = resultados_MINMIN[instancia][1]

            nrg_aux = 100.0 - (resultados_pals_1[instancia][1] * 100.0 / min_minmin)
            if nrg_aux > nrg_total_improvement_best_1:
                nrg_total_improvement_best_1 = nrg_aux
            nrg_total_improvement_avg_1 = nrg_total_improvement_avg_1 + (100.0 - (resultados_pals_1[instancia][5] * 100.0 / min_minmin))

            nrg_aux = 100.0 - (resultados_pals_2[instancia][1] * 100.0 / min_minmin)
            if nrg_aux > nrg_total_improvement_best_2:
                nrg_total_improvement_best_2 = nrg_aux
            nrg_total_improvement_avg_2 = nrg_total_improvement_avg_2 + (100.0 - (resultados_pals_2[instancia][5] * 100.0 / min_minmin))
            
        model_desc = ""
        type_desc = ""
        
        if item_grupo[0] == 'A': model_desc = 'Ali \emph{et al.}'
        if item_grupo[0] == 'B': model_desc = 'Braun \emph{et al.}'
        if item_grupo[1] == 'hihi': type_desc = 'High High'
        if item_grupo[1] == 'hilo': type_desc = 'High Low'
        if item_grupo[1] == 'lohi': type_desc = 'Low High'
        if item_grupo[1] == 'lolo': type_desc = 'Low Low'
        
        if mk_best_1 < mk_total_improvement_best_1: mk_best_1 = mk_total_improvement_best_1
        if mk_best_2 < mk_total_improvement_best_2: mk_best_2 = mk_total_improvement_best_2
        if nrg_best_1 < nrg_total_improvement_best_1: nrg_best_1 = nrg_total_improvement_best_1
        if nrg_best_2 < nrg_total_improvement_best_2: nrg_best_2 = nrg_total_improvement_best_2        
        
        mk_avg_1 = mk_avg_1 + (mk_total_improvement_avg_1 / items)
        nrg_avg_1 = nrg_avg_1 + (nrg_total_improvement_avg_1 / items)

        mk_avg_2 = mk_avg_2 + (mk_total_improvement_avg_2 / items)
        nrg_avg_2 = nrg_avg_2 + (nrg_total_improvement_avg_2 / items)
        
        #print "%s & %s & %.1f \\%% & %.1f \\%% \\\\" % (model_desc, type_desc, mk_total_improvement_avg / items, nrg_total_improvement_avg / items)
        print "%s & %s & $%.1f \\%%$ & $%.1f \\%%$ & $%.1f \\%%$ & $%.1f \\%%$ & $%.1f \\%%$ & $%.1f \\%%$ & $%.1f \\%%$ & $%.1f \\%%$ \\\\" % (model_desc, type_desc, mk_total_improvement_best_1, mk_total_improvement_avg_1 / items, nrg_total_improvement_best_1, nrg_total_improvement_avg_1 / items, mk_total_improvement_best_2, mk_total_improvement_avg_2 / items, nrg_total_improvement_best_2, nrg_total_improvement_avg_2 / items)

    print "%s & $%.1f \\%%$ & $%.1f \\%%$ & $%.1f \\%%$ & $%.1f \\%%$ & $%.1f \\%%$ & $%.1f \\%%$ & $%.1f \\%%$ & $%.1f \\%%$ \\\\" % (dimension, mk_best_1, mk_avg_1 / len(instancias_grupo_2), nrg_best_1, nrg_avg_1 / len(instancias_grupo_2), mk_best_2, mk_avg_2 / len(instancias_grupo_2), nrg_best_2, nrg_avg_2 / len(instancias_grupo_2))

    csv = "cons,heter,improv. best makespan,improv. best energy,improv. avg makespan,improv. avg energy,avg std dev makespan,avg std dev energy,avg nd\n"

    #print "[====== CSV ======]"
    #print csv
