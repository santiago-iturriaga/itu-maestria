import sys
import os
import math

PATH_AGA = "metrics-aga/pals-aga"
PATH_ADHOC = "metrics-adhoc/pals-1"
ITERATIONS=30

#SCENARIOS=(0,3,6,9,10,11,13,14,16,17,19)
#WORKLOADS=("A.u_c_hihi","A.u_c_hilo","A.u_c_lohi","A.u_c_lolo","A.u_i_hihi","A.u_i_hilo","A.u_i_lohi","A.u_i_lolo","A.u_s_hihi","A.u_s_hilo","A.u_s_lohi","A.u_s_lolo","B.u_c_hihi","B.u_c_hilo","B.u_c_lohi","B.u_c_lolo","B.u_i_hihi","B.u_i_hilo","B.u_i_lohi","B.u_i_lolo","B.u_s_hihi","B.u_s_hilo","B.u_s_lohi","B.u_s_lolo")

#SCENARIOS=(9,)
#WORKLOADS=("A.u_c_hihi",)

SCENARIOS=(0,3,6,9,10,11,13,14,16,17,19)
#SCENARIOS=(0,6,9,11,13,16,17,19)
#SCENARIOS=(0,6,11,13,16,19)
WORKLOADS=("A.u_c_hihi","A.u_c_hilo","A.u_c_lohi","A.u_c_lolo","B.u_c_hihi","B.u_c_hilo","B.u_c_lohi","B.u_c_lolo")
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

    avg_nd = [0.0, 0.0]
    avg_std_nd = [0.0, 0.0]

    avg_spread = [0.0, 0.0]
    avg_std_spread = [0.0, 0.0]

    avg_igd = [0.0, 0.0]
    avg_std_igd = [0.0, 0.0]

    avg_hv = [0.0, 0.0]
    avg_std_hv = [0.0, 0.0]

    nd_aga = []
    nd_adhoc = []
    spread_aga = []
    spread_adhoc = []
    igd_aga = []
    igd_adhoc = []
    hv_aga = []
    hv_adhoc = []

    for s in SCENARIOS:
        for w in WORKLOADS:
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

            #print hv_aga

            aggr_hv_aga = aggr_value(hv_aga)
            aggr_hv_adhoc = aggr_value(hv_adhoc)

            igd_aga = get_metric(PATH_AGA, s, w, "igd")
            igd_adhoc = get_metric(PATH_ADHOC, s, w, "igd")

            aggr_igd_aga = aggr_value(igd_aga)
            aggr_igd_adhoc = aggr_value(igd_adhoc)

            #print "%s %s,%s,%s,%s,%s" % (s,w,aggr_nd_aga[0],aggr_nd_aga[1],aggr_nd_adhoc[0],aggr_nd_adhoc[1]),
            #print ",%s,%s,%s,%s" % (aggr_spread_aga[0],aggr_spread_aga[1],aggr_spread_adhoc[0],aggr_spread_adhoc[1]),
            #print ",%s,%s,%s,%s" % (aggr_hv_aga[0],aggr_hv_aga[1],aggr_hv_adhoc[0],aggr_hv_adhoc[1]),
            #print ",%s,%s,%s,%s" % (aggr_igd_aga[0],aggr_igd_aga[1],aggr_igd_adhoc[0],aggr_igd_adhoc[1])

            print "%s %s"%(s,w),
            print ",%s,%s,%s,%s" % (aggr_nd_aga[0],aggr_nd_aga[1],aggr_nd_adhoc[0],aggr_nd_adhoc[1]),

            min_spread = aggr_spread_aga[0]
            if aggr_spread_adhoc[0] < min_spread: min_spread = aggr_spread_adhoc[0]

            print ",%s,%s,%s,%s" % (aggr_spread_aga[0]/min_spread,aggr_spread_aga[1]/min_spread,aggr_spread_adhoc[0]/min_spread,aggr_spread_adhoc[1]/min_spread),

            max_hv = aggr_hv_aga[0]
            if aggr_hv_adhoc[0] > max_hv: max_hv = aggr_hv_adhoc[0]
            if max_hv == 0: max_hv = 1

            print ",%s,%s,%s,%s" % (aggr_hv_aga[0]/max_hv,aggr_hv_aga[1]/max_hv,aggr_hv_adhoc[0]/max_hv,aggr_hv_adhoc[1]/max_hv),

            min_igd = aggr_igd_aga[0]
            if aggr_igd_adhoc[0] < min_igd: min_igd = aggr_igd_adhoc[0]
            if min_igd == 0: min_igd = 1

            print ",%s,%s,%s,%s" % (aggr_igd_aga[0]/min_igd,aggr_igd_aga[1]/min_igd,aggr_igd_adhoc[0]/min_igd,aggr_igd_adhoc[1]/min_igd),
            print ""

            avg_nd[0] += aggr_nd_aga[0]
            avg_nd[1] += aggr_nd_adhoc[0]
            avg_std_nd[0] += aggr_nd_aga[1]
            avg_std_nd[1] += aggr_nd_adhoc[1]

            avg_spread[0] += aggr_spread_aga[0]/min_spread
            avg_spread[1] += aggr_spread_adhoc[0]/min_spread
            avg_std_spread[0] += aggr_spread_aga[1]/min_spread
            avg_std_spread[1] += aggr_spread_adhoc[1]/min_spread

            avg_igd[0] += aggr_igd_aga[0]/min_igd
            avg_igd[1] += aggr_igd_adhoc[0]/min_igd
            avg_std_igd[0] += aggr_igd_aga[1]/min_igd
            avg_std_igd[1] += aggr_igd_adhoc[1]/min_igd

            avg_hv[0] += aggr_hv_aga[0]/max_hv
            avg_hv[1] += aggr_hv_adhoc[0]/max_hv
            avg_std_hv[0] += aggr_hv_aga[1]/max_hv
            avg_std_hv[1] += aggr_hv_adhoc[1]/max_hv

            nd_aga.append(aggr_nd_aga[0])
            nd_adhoc.append(aggr_nd_adhoc[0])

            spread_aga.append(aggr_spread_aga[0])
            spread_adhoc.append(aggr_spread_adhoc[0])

            igd_aga.append(aggr_igd_aga[0])
            igd_adhoc.append(aggr_igd_adhoc[0])

            hv_aga.append(aggr_hv_aga[0])
            hv_adhoc.append(aggr_hv_adhoc[0])

    l = len(SCENARIOS)*len(WORKLOADS)
    avg_nd[0] = avg_nd[0]/l
    avg_nd[1] = avg_nd[1]/l
    avg_std_nd[0] = avg_std_nd[0]/l
    avg_std_nd[1] = avg_std_nd[1]/l

    avg_spread[0] = avg_spread[0]/l
    avg_spread[1] = avg_spread[1]/l
    avg_std_spread[0] = avg_std_spread[0]/l
    avg_std_spread[1] = avg_std_spread[1]/l

    avg_igd[0] = avg_igd[0]/l
    avg_igd[1] = avg_igd[1]/l
    avg_std_igd[0] = avg_std_igd[0]/l
    avg_std_igd[1] = avg_std_igd[1]/l

    avg_hv[0] = avg_hv[0]/l
    avg_hv[1] = avg_hv[1]/l
    avg_std_hv[0] = avg_std_hv[0]/l
    avg_std_hv[1] = avg_std_hv[1]/l

    nd_aga = aggr_value(nd_aga)
    nd_adhoc = aggr_value(nd_adhoc)

    spread_aga = aggr_value(spread_aga)
    spread_adhoc = aggr_value(spread_adhoc)

    print igd_aga

    igd_aga = aggr_value(igd_aga)
    igd_adhoc = aggr_value(igd_adhoc)

    print igd_aga

    hv_aga = aggr_value(hv_aga)
    hv_adhoc = aggr_value(hv_adhoc)

    min_spread = spread_aga[0]
    if spread_adhoc[0] < min_spread: min_spread = spread_adhoc[0]

    max_hv = hv_aga[0]
    if hv_adhoc[0] > max_hv: max_hv = hv_adhoc[0]

    min_igd = igd_aga[0]
    if igd_adhoc[0] < min_igd: min_igd = igd_adhoc[0]

    print ""
    print "$%s\pm%s$,$%s\pm%s$" % (nd_aga[0],nd_aga[1],nd_adhoc[0],nd_adhoc[1])
    print "$%.2f\pm%.2f$,$%.2f\pm%.2f$" % (spread_aga[0]/min_spread,spread_aga[1]/min_spread,spread_adhoc[0]/min_spread,spread_adhoc[1]/min_spread)
    print "%s,%s,%s,%s" % (igd_aga[0]/min_igd,igd_aga[1]/min_igd,igd_adhoc[0]/min_igd,igd_adhoc[1]/min_igd)
    print "%s,%s,%s,%s" % (hv_aga[0]/max_hv,hv_aga[1]/max_hv,hv_adhoc[0]/max_hv,hv_adhoc[1]/max_hv)
    #print ""
    #print "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" % (nd_aga[0],nd_aga[1],nd_adhoc[0],nd_adhoc[1],spread_aga[0]/min_spread,spread_aga[1]/min_spread,spread_adhoc[0]/min_spread,spread_adhoc[1]/min_spread,igd_aga[0]/min_igd,igd_aga[1]/min_igd,igd_adhoc[0]/min_igd,igd_adhoc[1]/min_igd,hv_aga[0]/max_hv,hv_aga[1]/max_hv,hv_adhoc[0]/max_hv,hv_adhoc[1]/max_hv)
