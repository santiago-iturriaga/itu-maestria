import sys
import os
import math

PATH_FP = "metrics-fp/fp"
PATH_AGA = "metrics-aga/pals-aga"
PATH_ADHOC = "metrics-adhoc/pals-1"
ITERATIONS=30

SCENARIOS=(0,3,6,9,10,11,13,14,16,17,19)
#SCENARIOS=(0,)
WORKLOADS=("A.u_c_hihi","A.u_c_hilo","A.u_c_lohi","A.u_c_lolo","A.u_i_hihi","A.u_i_hilo","A.u_i_lohi","A.u_i_lolo","A.u_s_hihi","A.u_s_hilo","A.u_s_lohi","A.u_s_lolo","B.u_c_hihi","B.u_c_hilo","B.u_c_lohi","B.u_c_lolo","B.u_i_hihi","B.u_i_hilo","B.u_i_lohi","B.u_i_lolo","B.u_s_hihi","B.u_s_hilo","B.u_s_lohi","B.u_s_lolo")
#WORKLOADS=("B.u_c_hihi",)

#SCENARIOS=(9,)
#WORKLOADS=("A.u_c_hihi",)

#SCENARIOS=(0,3,6,9,10,11,13,14,16,17,19)
#SCENARIOS=(0,6,9,11,13,16,17,19)
#SCENARIOS=(0,6,11,13,16,19)

#WORKLOADS=("B.u_c_lohi",)
#WORKLOADS=("A.u_c_hihi","A.u_c_hilo","A.u_c_lohi","A.u_c_lolo","B.u_c_hihi","B.u_c_hilo","B.u_c_lohi","B.u_c_lolo")
#WORKLOADS=("A.u_c_hihi","A.u_c_hilo","A.u_c_lohi","A.u_c_lolo","A.u_s_hihi","A.u_s_hilo","A.u_s_lohi","A.u_s_lolo","B.u_c_hihi","B.u_c_hilo","B.u_c_lohi","B.u_c_lolo","B.u_s_hihi","B.u_s_hilo","B.u_s_lohi","B.u_s_lolo")

def aggr_value(values):
    if len(values) > 0:
        total = 0.0
        for v in values:
            total += v

        mean = total / float(len(values))

        aux = 0.0
        for v in values:
            aux += pow(v-mean,2) / float(len(values)) #float(len(values)-1)

        return (mean,math.sqrt(aux))
    else:
        return (0,0)

def get_fp_metric(path,s,w,suffix):
    valor_aux = 0
    full_path = path + ".scenario." + str(s) + ".workload." + w + "." + suffix
    try:
        file = open(full_path)

        valor_aux = file.readline()
        valor = float(valor_aux.split(' ')[0])

        return (valor, 0)
    except:
        print full_path
        print valor_aux

    return (0,0)

def get_metric(path,s,w,suffix):
    valores = []

    for i in range(30):
        valor_aux = 0
        full_path = path + ".scenario." + str(s) + ".workload." + w + "." + str(i) + "." + suffix
        try:
            file = open(full_path)

            valor_aux = file.readline()
            valor = float(valor_aux.split(' ')[0])

            #if not math.isnan(valor):
            if valor > 0:
                valores.append(valor)
        except:
            print full_path
            print valor_aux

    return valores

if __name__ == '__main__':
    print "instance,aga nd avg,aga nd stdev,adhoc nd avg,adhoc nd stdev,aga spread avg,aga spread stdev,adhoc spread avg,adhoc spread stdev,aga hv avg,aga hv stdev,adhoc hv avg,adhoc hv stdev,aga igd avg,aga igd stdev,adhoc igd avg,adhoc igd stdev"

    nd_aga_w = {}
    nd_adhoc_w = {}
    spread_aga_w = {}
    spread_adhoc_w = {}
    igd_aga_w = {}
    igd_adhoc_w = {}
    hv_aga_w = {}
    hv_adhoc_w = {}

    for w in WORKLOADS:
        nd_aga_w[w] = []
        nd_adhoc_w[w] = []
        spread_aga_w[w] = []
        spread_adhoc_w[w] = []
        igd_aga_w[w] = []
        igd_adhoc_w[w] = []
        hv_aga_w[w] = []
        hv_adhoc_w[w] = []

        for s in SCENARIOS:
            #print "%s %s" % (w,s)

            nd_aga = get_metric(PATH_AGA, s, w, "nd")
            nd_adhoc = get_metric(PATH_ADHOC, s, w, "nd")

            spread_aga = get_metric(PATH_AGA, s, w, "spread")
            spread_adhoc = get_metric(PATH_ADHOC, s, w, "spread")

            hv_aga = get_metric(PATH_AGA, s, w, "hv")
            hv_adhoc = get_metric(PATH_ADHOC, s, w, "hv")

            igd_aga = get_metric(PATH_AGA, s, w, "igd")
            igd_adhoc = get_metric(PATH_ADHOC, s, w, "igd")

            avg_spread_aga = aggr_value(spread_aga)
            avg_spread_adhoc = aggr_value(spread_adhoc)

            avg_igd_aga = aggr_value(igd_aga)
            avg_igd_adhoc = aggr_value(igd_adhoc)

            max_spread = max(spread_aga)
            if max_spread > max(spread_adhoc): max_spread = max(spread_adhoc)

            min_spread = min(spread_aga)
            if min_spread > min(spread_adhoc): min_spread = min(spread_adhoc)

            min_igd = min(igd_aga)
            if min_igd > min(igd_adhoc): min_igd = min(igd_adhoc)

            max_igd = max(igd_aga)
            if max_igd < max(igd_adhoc): max_igd = max(igd_adhoc)

            for i in hv_aga:
                hv_aga_w[w].append(i)
            for i in hv_adhoc:
                hv_adhoc_w[w].append(i)

            for i in nd_aga:
                nd_aga_w[w].append(i)
            for i in nd_adhoc:
                nd_adhoc_w[w].append(i)

            #print min_spread
            #print nd_aga
            #print igd_adhoc

            #for i in spread_aga:
                #spread_aga_w[w].append((i-min_spread)/(max_spread-min_spread))
                #spread_aga_w[w].append(i)
            #for i in spread_adhoc:
                #spread_adhoc_w[w].append((i-min_spread)/(max_spread-min_spread))
                #spread_adhoc_w[w].append(i)

            spread_aga_w[w].append(avg_spread_aga[0]/max_spread)
            spread_adhoc_w[w].append(avg_spread_adhoc[0]/max_spread)

            #print spread_aga_w[w]

            #for i in igd_aga:
                #igd_aga_w[w].append((i-min_igd)/(max_igd-min_igd))
                #igd_aga_w[w].append(i/min_igd)
                #igd_aga_w[w].append(i/max_igd)
                #igd_aga_w[w].append(i)
            #for i in igd_adhoc:
                #igd_adhoc_w[w].append((i-min_igd)/(max_igd-min_igd))
                #igd_adhoc_w[w].append(i/max_igd)
                #igd_adhoc_w[w].append(i/min_igd)
                #igd_adhoc_w[w].append(i)

            igd_aga_w[w].append(avg_igd_aga[0]/max_igd)
            igd_adhoc_w[w].append(avg_igd_adhoc[0]/max_igd)


    print "================"

    nd_aga_all = []
    nd_adhoc_all = []
    spread_aga_all = []
    spread_adhoc_all = []
    igd_aga_all = []
    igd_adhoc_all = []
    hv_aga_all = []
    hv_adhoc_all = []

    for w in WORKLOADS:
        #print w
        #print spread_aga_w[w]

        nd_aga_all.append(aggr_value(nd_aga_w[w])[0])
        nd_adhoc_all.append(aggr_value(nd_adhoc_w[w])[0])

        spread_aga_all.append(aggr_value(spread_aga_w[w])[0])
        spread_adhoc_all.append(aggr_value(spread_adhoc_w[w])[0])

        igd_aga_all.append(aggr_value(igd_aga_w[w])[0])
        igd_adhoc_all.append(aggr_value(igd_adhoc_w[w])[0])

        hv_aga_all.append(aggr_value(hv_aga_w[w])[0])
        hv_adhoc_all.append(aggr_value(hv_adhoc_w[w])[0])

    #print spread_aga_all

    (aga_avg_nd_mean,aga_avg_nd_stdev) = aggr_value(nd_aga_all)
    (adhoc_avg_nd_mean,adhoc_avg_nd_stdev) = aggr_value(nd_adhoc_all)

    (aga_avg_spread_mean,aga_avg_spread_stdev) = aggr_value(spread_aga_all)
    (adhoc_avg_spread_mean,adhoc_avg_spread_stdev) = aggr_value(spread_adhoc_all)

    (aga_avg_igd_mean,aga_avg_igd_stdev) = aggr_value(igd_aga_all)
    (adhoc_avg_igd_mean,adhoc_avg_igd_stdev) = aggr_value(igd_adhoc_all)

    (aga_avg_rhv_mean,aga_avg_rhv_stdev) = aggr_value(hv_aga_all)
    (adhoc_avg_rhv_mean,adhoc_avg_rhv_stdev) = aggr_value(hv_adhoc_all)

    print "NG/IGD"
    print "%.2f$\pm$%.2f & %.2f$\pm$%.2f" % (aga_avg_nd_mean,aga_avg_nd_stdev,adhoc_avg_nd_mean,adhoc_avg_nd_stdev),
    print "& %.2f$\pm$%.2f & %.2f$\pm$%.2f" % (aga_avg_igd_mean / min((aga_avg_igd_mean,adhoc_avg_igd_mean)),
        aga_avg_igd_stdev / min((aga_avg_igd_mean,adhoc_avg_igd_mean)), \
        adhoc_avg_igd_mean / min((aga_avg_igd_mean,adhoc_avg_igd_mean)), \
        adhoc_avg_igd_stdev / min((aga_avg_igd_mean,adhoc_avg_igd_mean))),
    print "\\\\"

    print "Spread/HV"
    print "%.2f$\pm$%.2f & %.2f$\pm$%.2f" % (aga_avg_spread_mean / min((aga_avg_spread_mean,adhoc_avg_spread_mean)), \
        aga_avg_spread_stdev / min((aga_avg_spread_mean,adhoc_avg_spread_mean)), \
        adhoc_avg_spread_mean / min((aga_avg_spread_mean,adhoc_avg_spread_mean)), \
        adhoc_avg_spread_stdev / min((aga_avg_spread_mean,adhoc_avg_spread_mean))),
    print "& %.2f$\pm$%.2f & %.2f\pm%.2f" % (aga_avg_rhv_mean,aga_avg_rhv_stdev,adhoc_avg_rhv_mean,adhoc_avg_rhv_stdev),
    print "\\\\"

    print "==========================================="

    for w in WORKLOADS:
        nd_aga = aggr_value(nd_aga_w[w])
        nd_adhoc = aggr_value(nd_adhoc_w[w])

        spread_aga = aggr_value(spread_aga_w[w])
        spread_adhoc = aggr_value(spread_adhoc_w[w])

        igd_aga = aggr_value(igd_aga_w[w])
        igd_adhoc = aggr_value(igd_adhoc_w[w])

        hv_aga = aggr_value(hv_aga_w[w])
        hv_adhoc = aggr_value(hv_adhoc_w[w])

        #min_spread = spread_aga[0]
        #if spread_adhoc[0] < min_spread: min_spread = spread_adhoc[0]

        min_igd = igd_aga[0]
        if igd_adhoc[0] < min_igd: min_igd = igd_adhoc[0]

        print w

        print "ND/IGD"
        print "%.2f$\pm$%.2f & %.2f$\pm$%.2f" % (nd_aga[0],nd_aga[1],nd_adhoc[0],nd_adhoc[1]),
        print "& %.2f$\pm$%.2f & %.2f$\pm$%.2f" % (igd_aga[0]/min_igd,igd_aga[1]/min_igd,igd_adhoc[0]/min_igd,igd_adhoc[1]/min_igd),
        #print "& %.2f$\pm$%.2f & %.2f$\pm$%.2f" % (igd_aga[0],igd_aga[1],igd_adhoc[0],igd_adhoc[1]),
        print "\\\\"

        print ""

    print "==========================================="

    for w in WORKLOADS:
        nd_aga = aggr_value(nd_aga_w[w])
        nd_adhoc = aggr_value(nd_adhoc_w[w])

        spread_aga = aggr_value(spread_aga_w[w])
        spread_adhoc = aggr_value(spread_adhoc_w[w])

        igd_aga = aggr_value(igd_aga_w[w])
        igd_adhoc = aggr_value(igd_adhoc_w[w])

        hv_aga = aggr_value(hv_aga_w[w])
        hv_adhoc = aggr_value(hv_adhoc_w[w])

        min_spread = spread_aga[0]
        if spread_adhoc[0] < min_spread: min_spread = spread_adhoc[0]

        #min_igd = igd_aga[0]
        #if igd_adhoc[0] < min_igd: min_igd = igd_adhoc[0]

        print w

        print "Spread/HV"
        print "%.2f$\pm$%.2f & %.2f$\pm$%.2f" % (spread_aga[0]/min_spread,spread_aga[1]/min_spread,spread_adhoc[0]/min_spread,spread_adhoc[1]/min_spread),
        #print "%.2f$\pm$%.2f & %.2f$\pm$%.2f" % (spread_aga[0],spread_aga[1],spread_adhoc[0],spread_adhoc[1]),
        print "& %.2f$\pm$%.2f & %.2f$\pm$%.2f" % (hv_aga[0],hv_aga[1],hv_adhoc[0],hv_adhoc[1]),
        print "\\\\"

        print ""

    kw_data = {}
    kw_data_alg = {'nd':{'c':[],'s':[],'i':[]}, \
        'igd':{'c':[],'s':[],'i':[]}, \
        'spread':{'c':[],'s':[],'i':[]}, \
        'hv':{'c':[],'s':[],'i':[]}}
    with open("kw.txt") as f:
        for line in f:
            raw = line.strip().split("|")

            #print "(%s)" % raw

            if not raw[0] in kw_data:
                kw_data[raw[0]] = {}

            kw_data[raw[0]][raw[1]] = raw[2]
            kw_data_alg[raw[0]][raw[1]] = (int(raw[3]),int(raw[4]))

    Latex_Q = ""
    Latex_D = ""

    kw_aga_count ={'nd':0,'igd':0,'hv':0,'spread':0}
    kw_adhoc_count = {'nd':0,'igd':0,'hv':0,'spread':0}

    for w in WORKLOADS:
        nd_aga = aggr_value(nd_aga_w[w])
        nd_adhoc = aggr_value(nd_adhoc_w[w])

        spread_aga = aggr_value(spread_aga_w[w])
        spread_adhoc = aggr_value(spread_adhoc_w[w])

        igd_aga = aggr_value(igd_aga_w[w])
        igd_adhoc = aggr_value(igd_adhoc_w[w])

        hv_aga = aggr_value(hv_aga_w[w])
        hv_adhoc = aggr_value(hv_adhoc_w[w])

        min_spread = spread_aga[0]
        if spread_adhoc[0] < min_spread: min_spread = spread_adhoc[0]

        min_igd = igd_aga[0]
        if igd_adhoc[0] < min_igd: min_igd = igd_adhoc[0]

        model_desc = ""
        type_desc = ""
        heter_desc = ""
        if w[0] == 'A': model_desc = 'Ali'
        if w[0] == 'B': model_desc = 'Braun'
        if w[4] == 'c': type_desc = 'cons.'
        if w[4] == 's': type_desc = 'semi.'
        if w[4] == 'i': type_desc = 'incons.'
        if w[6:] == 'hihi': heter_desc = 'high high'
        if w[6:] == 'hilo': heter_desc = 'high low'
        if w[6:] == 'lohi': heter_desc = 'low high'
        if w[6:] == 'lolo': heter_desc = 'low low'

        if model_desc == 'Braun' and type_desc == 'cons.' and heter_desc == 'high high':
            Latex_Q = Latex_Q + "\\cline{2-12}\n"
            Latex_D = Latex_D + "\\cline{2-12}\n"

        if type_desc != 'cons.' and heter_desc == 'high high':
            Latex_Q = Latex_Q + "\\cline{3-4}\\cline{6-8}\\cline{10-12}\n"
            Latex_D = Latex_D + "\\cline{3-4}\\cline{6-8}\\cline{10-12}\n"

        if type_desc == 'cons.' and heter_desc == 'high high':
            Latex_Q = Latex_Q + "& \multirow{12}{*}{%s} & " % (model_desc)
            Latex_D = Latex_D + "& \multirow{12}{*}{%s} & " % (model_desc)
        else:
            Latex_Q = Latex_Q + " & & "
            Latex_D = Latex_D + " & & "

        if heter_desc == 'high high':
            Latex_Q = Latex_Q + "\multirow{4}{*}{%s} & " % (type_desc)
            Latex_D = Latex_D + "\multirow{4}{*}{%s} & " % (type_desc)
        else:
            Latex_Q = Latex_Q + " & "
            Latex_D = Latex_D + " & "

        Latex_Q = Latex_Q + "%s & & %.2f$\pm$%.2f & %.2f$\pm$%.2f & %s " % (heter_desc, \
            nd_aga[0],nd_aga[1],nd_adhoc[0],nd_adhoc[1], kw_data['nd'][w])

        Latex_Q = Latex_Q + "& & %.2f$\pm$%.2f & %.2f$\pm$%.2f & %s \\\\ \n" % (igd_aga[0]/min_igd, \
            igd_aga[1]/min_igd,igd_adhoc[0]/min_igd,igd_adhoc[1]/min_igd, kw_data['igd'][w])

        #Latex_Q = Latex_Q + "& & %.2f$\pm$%.2f & %.2f$\pm$%.2f & %s \\\\ \n" % (igd_aga[0], \
        #    igd_aga[1],igd_adhoc[0],igd_adhoc[1], kw_data['igd'][w])

        Latex_D = Latex_D + "%s & & %.2f$\pm$%.2f & %.2f$\pm$%.2f & %s " % (heter_desc, \
            spread_aga[0]/min_spread,spread_aga[1]/min_spread,spread_adhoc[0]/min_spread,spread_adhoc[1]/min_spread, kw_data['spread'][w])

        #Latex_D = Latex_D + "%s & & %.2f$\pm$%.2f & %.2f$\pm$%.2f & %s " % (heter_desc, \
        #    spread_aga[0]/,spread_aga[1],spread_adhoc[0],spread_adhoc[1], kw_data['spread'][w])

        Latex_D = Latex_D + "& & %.2f$\pm$%.2f & %.2f$\pm$%.2f & %s \\\\ \n" % (hv_aga[0], \
            hv_aga[1],hv_adhoc[0],hv_adhoc[1], kw_data['hv'][w])
            
        #print kw_aga_count['nd'][w[4]]
        #print kw_data_alg['hv'][w][0]
        
        kw_aga_count['nd'] = kw_aga_count['nd'] + kw_data_alg['nd'][w][0]
        kw_adhoc_count['nd'] = kw_adhoc_count['nd'] + kw_data_alg['nd'][w][1]
        kw_aga_count['igd'] = kw_aga_count['igd'] + kw_data_alg['igd'][w][0]
        kw_adhoc_count['igd'] = kw_adhoc_count['igd'] + kw_data_alg['igd'][w][1]
        kw_aga_count['spread'] = kw_aga_count['spread'] + kw_data_alg['spread'][w][0]
        kw_adhoc_count['spread'] = kw_adhoc_count['spread'] + kw_data_alg['spread'][w][1]
        kw_aga_count['hv'] = kw_aga_count['hv'] + kw_data_alg['hv'][w][0]
        kw_adhoc_count['hv'] = kw_adhoc_count['hv'] + kw_data_alg['hv'][w][1]

    print kw_aga_count
    print kw_adhoc_count
    
    print "[========== ND/IGD ==========]"
    print Latex_Q
    print "\n\n[========== Spread/HV ==========]"
    print Latex_D

    t_nd_adhoc = {}
    t_nd_aga = {}
    t_nd_aga['c'] = []
    t_nd_adhoc['c'] = []
    t_nd_aga['i'] = []
    t_nd_adhoc['i'] = []
    t_nd_aga['s'] = []
    t_nd_adhoc['s'] = []

    t_spread_aga = {}
    t_spread_adhoc = {}
    t_spread_aga['c'] = []
    t_spread_adhoc['c'] = []
    t_spread_aga['i'] = []
    t_spread_adhoc['i'] = []
    t_spread_aga['s'] = []
    t_spread_adhoc['s'] = []

    t_igd_aga = {}
    t_igd_adhoc = {}
    t_igd_aga['c'] = []
    t_igd_adhoc['c'] = []
    t_igd_aga['i'] = []
    t_igd_adhoc['i'] = []
    t_igd_aga['s'] = []
    t_igd_adhoc['s'] = []

    t_hv_aga = {}
    t_hv_adhoc = {}
    t_hv_aga['c'] = []
    t_hv_adhoc['c'] = []
    t_hv_aga['i'] = []
    t_hv_adhoc['i'] = []
    t_hv_aga['s'] = []
    t_hv_adhoc['s'] = []

    for w in WORKLOADS:
        for i in nd_aga_w[w]:
            t_nd_aga[w[4]].append(i)
        for i in nd_adhoc_w[w]:
            t_nd_adhoc[w[4]].append(i)

        for i in spread_aga_w[w]:
            t_spread_aga[w[4]].append(i)
        for i in spread_adhoc_w[w]:
            t_spread_adhoc[w[4]].append(i)

        for i in igd_aga_w[w]:
            t_igd_aga[w[4]].append(i)
        for i in igd_adhoc_w[w]:
            t_igd_adhoc[w[4]].append(i)

        for i in hv_aga_w[w]:
            t_hv_aga[w[4]].append(i)
        for i in hv_adhoc_w[w]:
            t_hv_adhoc[w[4]].append(i)

    Latex_Q = ""
    Latex_D = ""

    for t in ('c','s','i'):
        nd_aga = aggr_value(t_nd_aga[t])
        nd_adhoc = aggr_value(t_nd_adhoc[t])

        spread_aga = aggr_value(t_spread_aga[t])
        spread_adhoc = aggr_value(t_spread_adhoc[t])

        igd_aga = aggr_value(t_igd_aga[t])
        igd_adhoc = aggr_value(t_igd_adhoc[t])

        hv_aga = aggr_value(t_hv_aga[t])
        hv_adhoc = aggr_value(t_hv_adhoc[t])

        min_spread = spread_aga[0]
        if spread_adhoc[0] < min_spread: min_spread = spread_adhoc[0]

        min_igd = igd_aga[0]
        if igd_adhoc[0] < min_igd: min_igd = igd_adhoc[0]

        type_desc = ""
        if t == 'c': type_desc = 'cons.'
        if t == 's': type_desc = 'semi.'
        if t == 'i': type_desc = 'incons.'

        #print "(%s)" % kw_data['nd'][w]

        if kw_data['nd'][w] == "\textbf{AGA 11/11}":
            Latex_Q = Latex_Q + "%s & & \textbf{%.2f$\pm$%.2f} & %.2f$\pm$%.2f" % (type_desc, \
                nd_aga[0],nd_aga[1],nd_adhoc[0],nd_adhoc[1])
        elif kw_data['nd'][w] == "\textbf{FGAA 11/11}":
            Latex_Q = Latex_Q + "%s & & %.2f$\pm$%.2f & \textbf{%.2f$\pm$%.2f}" % (type_desc, \
                nd_aga[0],nd_aga[1],nd_adhoc[0],nd_adhoc[1])
        else:
            Latex_Q = Latex_Q + "%s & & %.2f$\pm$%.2f & %.2f$\pm$%.2f" % (type_desc, \
                nd_aga[0],nd_aga[1],nd_adhoc[0],nd_adhoc[1])

        if kw_data['igd'][w] == "\textbf{AGA 11/11}":
            Latex_Q = Latex_Q + "& & \textbf{%.2f$\pm$%.2f} & %.2f$\pm$%.2f \\\\ \n" % (igd_aga[0]/min_igd, \
                igd_aga[1]/min_igd,igd_adhoc[0]/min_igd,igd_adhoc[1]/min_igd)
        elif kw_data['igd'][w] == "\textbf{FGAA 11/11}":
            Latex_Q = Latex_Q + "& & %.2f$\pm$%.2f & \textbf{%.2f$\pm$%.2f} \\\\ \n" % (igd_aga[0]/min_igd, \
                igd_aga[1]/min_igd,igd_adhoc[0]/min_igd,igd_adhoc[1]/min_igd)
        else:
            Latex_Q = Latex_Q + "& & %.2f$\pm$%.2f & %.2f$\pm$%.2f \\\\ \n" % (igd_aga[0]/min_igd, \
                igd_aga[1]/min_igd,igd_adhoc[0]/min_igd,igd_adhoc[1]/min_igd)

        if kw_data['spread'][w] == "\textbf{AGA 11/11}":
            Latex_D = Latex_D + "%s & & \textbf{%.2f$\pm$%.2f} & %.2f$\pm$%.2f " % (type_desc, \
                spread_aga[0]/min_spread,spread_aga[1]/min_spread,spread_adhoc[0]/min_spread,spread_adhoc[1]/min_spread)
        elif kw_data['spread'][w] == "\textbf{FGAA 11/11}":
            Latex_D = Latex_D + "%s & & %.2f$\pm$%.2f & \textbf{%.2f$\pm$%.2f} " % (type_desc, \
                spread_aga[0]/min_spread,spread_aga[1]/min_spread,spread_adhoc[0]/min_spread,spread_adhoc[1]/min_spread)
        else:
            Latex_D = Latex_D + "%s & & %.2f$\pm$%.2f & %.2f$\pm$%.2f " % (type_desc, \
                spread_aga[0]/min_spread,spread_aga[1]/min_spread,spread_adhoc[0]/min_spread,spread_adhoc[1]/min_spread)

        if kw_data['hv'][w] == "\textbf{AGA 11/11}":
            Latex_D = Latex_D + "& & \textbf{%.2f$\pm$%.2f} & %.2f$\pm$%.2f \\\\ \n" % (hv_aga[0], \
                hv_aga[1],hv_adhoc[0],hv_adhoc[1])
        elif kw_data['hv'][w] == "\textbf{FGAA 11/11}":
            Latex_D = Latex_D + "& & %.2f$\pm$%.2f & \textbf{%.2f$\pm$%.2f} \\\\ \n" % (hv_aga[0], \
                hv_aga[1],hv_adhoc[0],hv_adhoc[1])
        else:
            Latex_D = Latex_D + "& & %.2f$\pm$%.2f & %.2f$\pm$%.2f \\\\ \n" % (hv_aga[0], \
                hv_aga[1],hv_adhoc[0],hv_adhoc[1])
                
    print "[========== ND/IGD ==========]"
    print Latex_Q
    print "\n\n[========== Spread/HV ==========]"
    print Latex_D
