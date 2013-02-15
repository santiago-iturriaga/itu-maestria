import sys
import os
import math

PATH_FP = "metrics-fp/fp"
PATH_AGA = "metrics-aga/pals-aga"
PATH_ADHOC = "metrics-adhoc/pals-1"
ITERATIONS=30

SCENARIOS=(0,3,6,9,10,11,13,14,16,17,19)
WORKLOADS=("A.u_c_hihi","A.u_c_hilo","A.u_c_lohi","A.u_c_lolo","A.u_i_hihi","A.u_i_hilo","A.u_i_lohi","A.u_i_lolo","A.u_s_hihi","A.u_s_hilo","A.u_s_lohi","A.u_s_lolo","B.u_c_hihi","B.u_c_hilo","B.u_c_lohi","B.u_c_lolo","B.u_i_hihi","B.u_i_hilo","B.u_i_lohi","B.u_i_lolo","B.u_s_hihi","B.u_s_hilo","B.u_s_lohi","B.u_s_lolo")

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

    nd_aga_g = []
    nd_adhoc_g = []
    spread_aga_g = []
    spread_adhoc_g = []
    igd_aga_g = []
    igd_adhoc_g = []
    hv_aga_g = []
    hv_adhoc_g = []

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
            nd_aga = get_metric(PATH_AGA, s, w, "nd")
            nd_adhoc = get_metric(PATH_ADHOC, s, w, "nd")

            aggr_nd_aga = aggr_value(nd_aga)
            aggr_nd_adhoc = aggr_value(nd_adhoc)

            spread_aga = get_metric(PATH_AGA, s, w, "spread")
            spread_adhoc = get_metric(PATH_ADHOC, s, w, "spread")

            aggr_spread_aga = aggr_value(spread_aga)
            aggr_spread_adhoc = aggr_value(spread_adhoc)

            hv_aga = get_metric(PATH_AGA, s, w, "hv")
            hv_adhoc = get_metric(PATH_ADHOC, s, w, "hv")
            
            aggr_hv_fp = get_fp_metric(PATH_FP, s, w, "hv")

            aggr_hv_aga = aggr_value(hv_aga)
            aggr_hv_adhoc = aggr_value(hv_adhoc)

            igd_aga = get_metric(PATH_AGA, s, w, "igd")
            igd_adhoc = get_metric(PATH_ADHOC, s, w, "igd")

            aggr_igd_aga = aggr_value(igd_aga)
            aggr_igd_adhoc = aggr_value(igd_adhoc)

            min_spread = aggr_spread_aga[0]
            if aggr_spread_adhoc[0] < min_spread: min_spread = aggr_spread_adhoc[0]
            if min_spread == 0: min_spread = 1

            max_hv = aggr_hv_fp[0] #aggr_hv_aga[0]

            #print "%s %s" % (w, s)
            #if max_hv == 0: max_hv = 1
            #print max_hv
            #print hv_adhoc
            #for i in hv_aga:
                #hv_aga_w[w].append(i/max_hv)
                #hv_aga_g.append(i/max_hv)
            #for i in hv_adhoc:
                #hv_adhoc_w[w].append(i/max_hv)
                #hv_adhoc_g.append(i/max_hv)
                    
            #if max_hv > 0 and aggr_hv_aga[0] > 0 and aggr_hv_adhoc[0] > 0:
            if max_hv > 0:
                #print aggr_hv_fp
                #print hv_adhoc
                if aggr_hv_aga[0] > 0:
                    for i in hv_aga:
                        hv_aga_w[w].append(i/max_hv)
                        hv_aga_g.append(i/max_hv)
                if aggr_hv_adhoc[0] > 0:
                    for i in hv_adhoc:
                        hv_adhoc_w[w].append(i/max_hv)
                        hv_adhoc_g.append(i/max_hv)
                #print hv_adhoc_w[w]

            min_igd = aggr_igd_aga[0]
            if aggr_igd_adhoc[0] < min_igd: min_igd = aggr_igd_adhoc[0]
            if min_igd == 0: min_igd = 1

            for i in nd_aga:
                nd_aga_w[w].append(i)
                nd_aga_g.append(i)
            for i in spread_aga:
                spread_aga_w[w].append(i/min_spread)
                spread_aga_g.append(i/min_spread)
            for i in igd_aga:
                igd_aga_w[w].append(i/min_igd)
                igd_aga_g.append(i/min_igd)

            for i in nd_adhoc:
                nd_adhoc_g.append(i)
                nd_adhoc_w[w].append(i)
            for i in spread_adhoc:
                spread_adhoc_w[w].append(i/min_spread)
                spread_adhoc_g.append(i/min_spread)
            for i in igd_adhoc:
                igd_adhoc_w[w].append(i/min_igd)
                igd_adhoc_g.append(i/min_igd)

    print "================"

    nd_aga = aggr_value(nd_aga_g)
    nd_adhoc = aggr_value(nd_adhoc_g)

    spread_aga = aggr_value(spread_aga_g)
    spread_adhoc = aggr_value(spread_adhoc_g)

    igd_aga = aggr_value(igd_aga_g)
    igd_adhoc = aggr_value(igd_adhoc_g)

    hv_aga = aggr_value(hv_aga_g)
    hv_adhoc = aggr_value(hv_adhoc_g)

    min_spread = spread_aga[0]
    if spread_adhoc[0] < min_spread: min_spread = spread_adhoc[0]

    max_hv = hv_aga[0]
    if hv_adhoc[0] > max_hv: max_hv = hv_adhoc[0]

    min_igd = igd_aga[0]
    if igd_adhoc[0] < min_igd: min_igd = igd_adhoc[0]

    #print ""
    #print "$%.2f\pm%.2f$ & $%.2f\pm%.2f$" % (nd_aga[0],nd_aga[1],nd_adhoc[0],nd_adhoc[1]),
    #print "& $%.2f\pm%.2f$ & $%.2f\pm%.2f$" % (spread_aga[0]/min_spread,spread_aga[1]/min_spread,spread_adhoc[0]/min_spread,spread_adhoc[1]/min_spread),
    #print "& $%.2f\pm%.2f$ & $%.2f\pm%.2f$" % (igd_aga[0]/min_igd,igd_aga[1]/min_igd,igd_adhoc[0]/min_igd,igd_adhoc[1]/min_igd),
    #print "& $%.2f\pm%.2f$ & $%.2f\pm%.2f$" % (hv_aga[0]/max_hv,hv_aga[1]/max_hv,hv_adhoc[0]/max_hv,hv_adhoc[1]/max_hv),
    #print "\\\\"
    
    print "NG/IGD"
    print "$%.2f\pm%.2f$ & $%.2f\pm%.2f$" % (nd_aga[0],nd_aga[1],nd_adhoc[0],nd_adhoc[1]),
    print "& $%.2f\pm%.2f$ & $%.2f\pm%.2f$" % (igd_aga[0]/min_igd,igd_aga[1]/min_igd,igd_adhoc[0]/min_igd,igd_adhoc[1]/min_igd),
    print "\\\\"
    
    print "Spread/HV"
    print "$%.2f\pm%.2f$ & $%.2f\pm%.2f$" % (spread_aga[0]/min_spread,spread_aga[1]/min_spread,spread_adhoc[0]/min_spread,spread_adhoc[1]/min_spread),
    #print "& $%.2f\pm%.2f$ & $%.2f\pm%.2f$" % (hv_aga[0]/max_hv,hv_aga[1]/max_hv,hv_adhoc[0]/max_hv,hv_adhoc[1]/max_hv),
    print "& $%.2f\pm%.2f$ & $%.2f\pm%.2f$" % (hv_aga[0],hv_aga[1],hv_adhoc[0],hv_adhoc[1]),
    print "\\\\"
    
    #print "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" % (nd_aga[0],nd_aga[1],nd_adhoc[0],nd_adhoc[1],spread_aga[0]/min_spread,spread_aga[1]/min_spread,spread_adhoc[0]/min_spread,spread_adhoc[1]/min_spread,igd_aga[0]/min_igd,igd_aga[1]/min_igd,igd_adhoc[0]/min_igd,igd_adhoc[1]/min_igd,hv_aga[0]/max_hv,hv_aga[1]/max_hv,hv_adhoc[0]/max_hv,hv_adhoc[1]/max_hv)

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

        max_hv = hv_aga[0]
        if hv_adhoc[0] > max_hv: max_hv = hv_adhoc[0]

        min_igd = igd_aga[0]
        if igd_adhoc[0] < min_igd: min_igd = igd_adhoc[0]

        #print ""
        #print "$%.2f\pm%.2f$ & $%.2f\pm%.2f$" % (nd_aga[0],nd_aga[1],nd_adhoc[0],nd_adhoc[1]),
        #print "& $%.2f\pm%.2f$ & $%.2f\pm%.2f$" % (spread_aga[0]/min_spread,spread_aga[1]/min_spread,spread_adhoc[0]/min_spread,spread_adhoc[1]/min_spread),
        #print "& $%.2f\pm%.2f$ & $%.2f\pm%.2f$" % (igd_aga[0]/min_igd,igd_aga[1]/min_igd,igd_adhoc[0]/min_igd,igd_adhoc[1]/min_igd),
        #print "& $%.2f\pm%.2f$ & $%.2f\pm%.2f$" % (hv_aga[0]/max_hv,hv_aga[1]/max_hv,hv_adhoc[0]/max_hv,hv_adhoc[1]/max_hv),
        #print "\\\\"
        
        print w
        
        print "ND/IGD"
        print "%.2f$\pm$%.2f & %.2f$\pm$%.2f" % (nd_aga[0],nd_aga[1],nd_adhoc[0],nd_adhoc[1]),
        print "& %.2f$\pm$%.2f & %.2f$\pm$%.2f" % (igd_aga[0]/min_igd,igd_aga[1]/min_igd,igd_adhoc[0]/min_igd,igd_adhoc[1]/min_igd),
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

        min_igd = igd_aga[0]
        if igd_adhoc[0] < min_igd: min_igd = igd_adhoc[0]
        
        print w
        
        print "Spread/HV"
        print "%.2f$\pm$%.2f & %.2f$\pm$%.2f" % (spread_aga[0]/min_spread,spread_aga[1]/min_spread,spread_adhoc[0]/min_spread,spread_adhoc[1]/min_spread),
        print "& %.2f$\pm$%.2f & %.2f$\pm$%.2f" % (hv_aga[0],hv_aga[1],hv_adhoc[0],hv_adhoc[1]),
        print "\\\\"

        print ""

    kw_data = {}
    with open("kw.txt") as f:
        for line in f:
            raw = line.strip().split("|")
            
            if not raw[0] in kw_data:
                kw_data[raw[0]] = {}
                
            kw_data[raw[0]][raw[1]] = raw[2]
    
    Latex_Q = ""
    Latex_D = ""

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

        max_hv = hv_aga[0]
        if hv_adhoc[0] > max_hv: max_hv = hv_adhoc[0]

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

        print w
        print w[0]
        print w[4]
        print w[6:]

        if model_desc == 'Braun' and type_desc == 'cons.' and heter_desc == 'high high':
            Latex_Q = Latex_Q + "\\hline\n"
            Latex_D = Latex_D + "\\hline\n"

        if type_desc != 'cons.' and heter_desc == 'high high':
            Latex_Q = Latex_Q + "\\cline{2-3}\\cline{5-7}\\cline{9-11}\n"
            Latex_D = Latex_D + "\\cline{2-3}\\cline{5-7}\\cline{9-11}\n"

        if type_desc == 'cons.' and heter_desc == 'high high':
            Latex_Q = Latex_Q + "\multirow{12}{*}{%s} & " % (model_desc)
            Latex_D = Latex_D + "\multirow{12}{*}{%s} & " % (model_desc)
        else:
            Latex_Q = Latex_Q + " & "
            Latex_D = Latex_D + " & "

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
            
        Latex_D = Latex_D + "%s & & %.2f$\pm$%.2f & %.2f$\pm$%.2f & %s " % (heter_desc, \
            spread_aga[0]/min_spread,spread_aga[1]/min_spread,spread_adhoc[0]/min_spread,spread_adhoc[1]/min_spread, kw_data['spread'][w])
            
        Latex_D = Latex_D + "& & %.2f$\pm$%.2f & %.2f$\pm$%.2f & %s \\\\ \n" % (hv_aga[0], \
            hv_aga[1],hv_adhoc[0],hv_adhoc[1], kw_data['hv'][w])

    print "[========== ND/IGD ==========]"
    print Latex_Q
    print "\n\n[========== Spread/HV ==========]"
    print Latex_D
