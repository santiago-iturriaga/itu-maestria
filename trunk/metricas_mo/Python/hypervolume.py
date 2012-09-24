#encoding:utf-8
"""
Calcula el hypervolumen de una colección de frentes
de pareto de funciones bi-objetivo.
"""

import kruskalwallis
import math

from decimal import *

chi2_g1 = ((0.05 ,3.84), (0.01 ,6.64), (0.001, 10.83))

"""
Calculate mean and standard deviation of data x[]:
    mean = {\sum_i x_i \over n}
    std = sqrt(\sum_i (x_i - mean)^2 \over n-1)
"""
def meanstdv(x):
    n, mean, std = len(x), 0, 0
    for a in x:
        mean = mean + a
    mean = mean / float(n)
    for a in x:
        std = std + (a - mean)**2
    std = math.sqrt(std / float(n-1))
    return mean, std
    
def mean(x):
    n, mean = len(x), 0
    for a in x:
        mean = mean + a
    mean = mean / float(n)
    return mean

def read_pf_file(f):
    points = []

    for line in open(f):
        tokens = line.strip().split(" ")

        if len(tokens)>1:
            #print tokens
            #print "%s %s" % (float(tokens[0]), float(tokens[len(tokens)-1]))
            points.append((float(tokens[0]), float(tokens[len(tokens)-1])))
        else:
            tokens = line.strip().strip("\t").split("\t")

            if len(tokens)>1:
                #print tokens
                #print "%s %s" % (float(tokens[0]), float(tokens[len(tokens)-1]))
                points.append((float(tokens[0]), float(tokens[len(tokens)-1])))

    return points

def get_w_point(true_pf, inst_pf):
    max_obj_1 = true_pf[0][0]
    max_obj_2 = true_pf[0][1]

    for p in true_pf:
        if p[0] > max_obj_1:
            max_obj_1 = p[0]
        if p[1] > max_obj_2:
            max_obj_2 = p[1]

    for p in inst_pf:
        if p[0] > max_obj_1:
            max_obj_1 = p[0]
        if p[1] > max_obj_2:
            max_obj_2 = p[1]

    return (max_obj_1 * 1.1, max_obj_2 * 1.1)
    #return (max_obj_1, max_obj_2)
    #return (float(5.420153), float(9293.50293))

def compute_hypervolume(true_pf, inst_pf):
    w = get_w_point(true_pf, inst_pf)

    #print "W point:"
    #print w

    true_pf = sorted(true_pf)
    inst_pf = sorted(inst_pf)

    #print true_pf
    #print inst_pf

    true_pf_hv = (w[0] - true_pf[0][0]) * (w[1] - true_pf[0][1])
    for p in range(len(true_pf)-1):
        hvol = (true_pf[p][1] - true_pf[p+1][1]) * (w[0] - true_pf[p+1][0])
        true_pf_hv = true_pf_hv + hvol

    inst_pf_hv = (w[0] - inst_pf[0][0]) * (w[1] - inst_pf[0][1])
    #print "(w[0] - inst_pf[0][0]) * (w[1] - inst_pf[0][1]) = (%.2f - %.2f) * (%.2f - %.2f) = %.2f" % (w[0], inst_pf[0][0], w[1], inst_pf[0][1], inst_pf_hv)
    for p in range(len(inst_pf)-1):
        hvol = (inst_pf[p][1] - inst_pf[p+1][1]) * (w[0] - inst_pf[p+1][0])
        #print "(inst_pf[p][1] - inst_pf[p+1][1]) * (w[0] - inst_pf[p+1][0]) = (%.2f - %.2f) * (%.2f - %.2f) = %.2f" % (inst_pf[p][1], inst_pf[p+1][1], w[0], inst_pf[p+1][0], hvol)
        inst_pf_hv = inst_pf_hv + hvol
        #print "%.2f" % (inst_pf_hv)

    #print "Hypervolume true PF: %s" % (true_pf_hv)
    #print "Hypervolume inst PF: %s (%s%%)" % (inst_pf_hv, inst_pf_hv / true_pf_hv * 100)

    if (true_pf_hv > 0):
        return inst_pf_hv / true_pf_hv
    else:
        return 0

def full_instancias(datos):
    (INSTANCES,SCENARIOS,WORKLOADS,TRUE_PF,BASE_PATH,FILE_PREFIX) = datos

    result_list = []

    for w in WORKLOADS:
        w_s_results = []
        full_results = []

        for s in SCENARIOS:
            true_pf_file = TRUE_PF+"/scenario."+str(s)+".workload."+w+".fp"

            #print "%s %s" % (w,s)
            i_results = []

            for i in range(INSTANCES):
                instance_pf_file = BASE_PATH+"/scenario."+str(s)+".workload."+w+"."+str(i)+"/"+FILE_PREFIX+"scenario."+str(s)+".workload."+w+".metrics"

                true_pf = read_pf_file(true_pf_file)
                instance_pf = read_pf_file(instance_pf_file)

                if len(true_pf) > 0 and len(instance_pf):
                    result = compute_hypervolume(true_pf, instance_pf)
                    if result > 0.1:
                        w_s_results.append(result)
                        i_results.append(result)
                        #print "%s %s %s = %.2f" % (w,s,i,result)

            full_results.append(i_results)

        mean = 0.0
        stdev = 0.0
        
        if w_s_results > 1:
            (mean, stdev) = meanstdv(w_s_results)
        
        result_list.append((mean,stdev, full_results))
        print "%s = %.2f +/- %.3f" % (w, mean, stdev)

    return result_list

def test():
    current_true_pf = read_pf_file("fp_0.txt")
    instance_true_pf = read_pf_file("fp_1.txt")

    compute_hypervolume(current_true_pf, instance_true_pf)

def test2():
    INSTANCES = 30
    SCENARIOS = (19,)
    #SCENARIOS = (3,6,11,16,17,19)
    #SCENARIOS = (0,3,6,9,10,11,13,14,16,17,19)
    WORKLOADS = ("A.u_c_lolo",)
    #WORKLOADS = sorted(("A.u_c_hihi","A.u_c_hilo","A.u_c_lohi","A.u_c_lolo","B.u_c_hihi","B.u_c_hilo","B.u_c_lohi","B.u_c_lolo"))

    DIMENSION = "512x16"
    print DIMENSION+" AGA"

    TRUE_PF = "/home/santiago/itu-maestria/svn/branches/palsCPU-Journal2/"+DIMENSION+".fp"
    #BASE_PATH = "/home/santiago/itu-maestria/svn/branches/palsCPU-Journal2/"+DIMENSION+".24_10s"
    BASE_PATH = "/home/santiago/itu-maestria/svn/branches/palsCPU-Journal2/"+DIMENSION+".24.adhoc"
    #FILE_PREFIX = "pals-aga."
    FILE_PREFIX = "pals-1."

    full_instancias((INSTANCES,SCENARIOS,WORKLOADS,TRUE_PF,BASE_PATH,FILE_PREFIX))

def full_dimensiones():
    INSTANCES = 30
    SCENARIOS = (0,3,6,9,10,11,13,14,16,17,19)
    WORKLOADS = sorted(("A.u_c_hihi","A.u_c_hilo","A.u_c_lohi","A.u_c_lolo","B.u_c_hihi","B.u_c_hilo","B.u_c_lohi","B.u_c_lolo"))

    DIMENSION = "512x16"
    print DIMENSION+" AGA"

    TRUE_PF = "/home/santiago/itu-maestria/svn/branches/palsCPU-Journal2/"+DIMENSION+".fp"
    BASE_PATH = "/home/santiago/itu-maestria/svn/branches/palsCPU-Journal2/"+DIMENSION+".24_10s"
    FILE_PREFIX = "pals-aga."

    aga = full_instancias((INSTANCES,SCENARIOS,WORKLOADS,TRUE_PF,BASE_PATH,FILE_PREFIX))

    print DIMENSION+" ADHOC"

    TRUE_PF = "/home/santiago/itu-maestria/svn/branches/palsCPU-Journal2/"+DIMENSION+".fp"
    BASE_PATH = "/home/santiago/itu-maestria/svn/branches/palsCPU-Journal2/"+DIMENSION+".24.adhoc"
    FILE_PREFIX = "pals-1."

    adhoc = full_instancias((INSTANCES,SCENARIOS,WORKLOADS,TRUE_PF,BASE_PATH,FILE_PREFIX))

    for i in range(len(WORKLOADS)):
        print WORKLOADS[i]

        best_aga = 0
        best_adhoc = 0

        for j in range(len(SCENARIOS)):
            #print str(SCENARIOS[j]) + " ",
            
            kw = kruskalwallis.kruskalwallis([aga[i][2][j],adhoc[i][2][j]], ignoreties = False)
        
            index = len(chi2_g1)-1
            while index >= 0 and kw < chi2_g1[index][1]:
                index = index - 1
            
            h0_rechazada = False
            if index >= 0:
                if kw > chi2_g1[index][1]:
                    h0_rechazada = True
            
            if h0_rechazada:
                #print "(se rechaza H0 con un p-value %f)" % (chi2_g1[index][0])
                
                if mean(aga[i][2][j]) > mean(adhoc[i][2][j]):
                    best_aga = best_aga + 1
                else:
                    best_adhoc = best_adhoc + 1
                
            else:
                #print "(no se puede rechazar H0)" 
                pass

        best_name = "none"
        best_count = ""
        if (best_aga > best_adhoc):
            best_name = "AGA"
            best_count = str(best_aga) + "/12"
        elif (best_aga < best_adhoc):
            best_name = "ad hoc"
            best_count = str(best_adhoc) + "/12"
        
        if best_aga == 12:
            print "& & \textbf{%.2f$\\pm$%.2f} & $%.2f\\pm%.2f$ & %s $%s$" % (aga[i][0], aga[i][1], adhoc[i][0], adhoc[i][1], best_name ,best_count)
        elif best_adhoc == 12:
            print "& & $%.2f\\pm%.2f$ & \textbf{%.2f$\\pm$%.2f} & %s $%s$" % (aga[i][0], aga[i][1], adhoc[i][0], adhoc[i][1], best_name ,best_count)
        else:
            print "& & $%.2f\\pm%.2f$ & $%.2f\\pm%.2f$ & %s $%s$" % (aga[i][0], aga[i][1], adhoc[i][0], adhoc[i][1], best_name ,best_count)

    # ==================================================================================

    DIMENSION = "1024x32"
    print DIMENSION+" AGA"

    TRUE_PF = "/home/santiago/itu-maestria/svn/branches/palsCPU-Journal2/"+DIMENSION+".fp"
    BASE_PATH = "/home/santiago/itu-maestria/svn/branches/palsCPU-Journal2/"+DIMENSION+".24_10s"
    FILE_PREFIX = "pals-aga."

    aga = full_instancias((INSTANCES,SCENARIOS,WORKLOADS,TRUE_PF,BASE_PATH,FILE_PREFIX))

    print DIMENSION+" ADHOC"

    TRUE_PF = "/home/santiago/itu-maestria/svn/branches/palsCPU-Journal2/"+DIMENSION+".fp"
    BASE_PATH = "/home/santiago/itu-maestria/svn/branches/palsCPU-Journal2/"+DIMENSION+".24.adhoc"
    FILE_PREFIX = "pals-1."

    adhoc = full_instancias((INSTANCES,SCENARIOS,WORKLOADS,TRUE_PF,BASE_PATH,FILE_PREFIX))

    for i in range(len(WORKLOADS)):
        print WORKLOADS[i]

        best_aga = 0
        best_adhoc = 0

        for j in range(len(SCENARIOS)):
            #print str(SCENARIOS[j]) + " ",
            
            kw = kruskalwallis.kruskalwallis([aga[i][2][j],adhoc[i][2][j]], ignoreties = False)
        
            index = len(chi2_g1)-1
            while index >= 0 and kw < chi2_g1[index][1]:
                index = index - 1
            
            h0_rechazada = False
            if index >= 0:
                if kw > chi2_g1[index][1]:
                    h0_rechazada = True
            
            if h0_rechazada:
                #print "(se rechaza H0 con un p-value %f)" % (chi2_g1[index][0])
                
                if mean(aga[i][2][j]) > mean(adhoc[i][2][j]):
                    best_aga = best_aga + 1
                else:
                    best_adhoc = best_adhoc + 1
                
            else:
                #print "(no se puede rechazar H0)" 
                pass

        best_name = "none"
        best_count = ""
        if (best_aga > best_adhoc):
            best_name = "AGA"
            best_count = str(best_aga) + "/12"
        elif (best_aga < best_adhoc):
            best_name = "ad hoc"
            best_count = str(best_adhoc) + "/12"
        
        if best_aga == 12:
            print "& & \textbf{%.2f$\\pm$%.2f} & $%.2f\\pm%.2f$ & %s $%s$" % (aga[i][0], aga[i][1], adhoc[i][0], adhoc[i][1], best_name ,best_count)
        elif best_adhoc == 12:
            print "& & $%.2f\\pm%.2f$ & \textbf{%.2f$\\pm$%.2f} & %s $%s$" % (aga[i][0], aga[i][1], adhoc[i][0], adhoc[i][1], best_name ,best_count)
        else:
            print "& & $%.2f\\pm%.2f$ & $%.2f\\pm%.2f$ & %s $%s$" % (aga[i][0], aga[i][1], adhoc[i][0], adhoc[i][1], best_name ,best_count)

    # ==================================================================================

    DIMENSION = "2048x64"
    print DIMENSION+" AGA"

    TRUE_PF = "/home/santiago/itu-maestria/svn/branches/palsCPU-Journal2/"+DIMENSION+".fp"
    BASE_PATH = "/home/santiago/itu-maestria/svn/branches/palsCPU-Journal2/"+DIMENSION+".24_10s"
    FILE_PREFIX = "pals-aga."

    aga = full_instancias((INSTANCES,SCENARIOS,WORKLOADS,TRUE_PF,BASE_PATH,FILE_PREFIX))

    print DIMENSION+" ADHOC"

    TRUE_PF = "/home/santiago/itu-maestria/svn/branches/palsCPU-Journal2/"+DIMENSION+".fp"
    BASE_PATH = "/home/santiago/itu-maestria/svn/branches/palsCPU-Journal2/"+DIMENSION+".24.adhoc"
    FILE_PREFIX = "pals-1."

    adhoc = full_instancias((INSTANCES,SCENARIOS,WORKLOADS,TRUE_PF,BASE_PATH,FILE_PREFIX))

    for i in range(len(WORKLOADS)):
        print WORKLOADS[i]

        empty = 0
        best_aga = 0
        best_adhoc = 0

        for j in range(len(SCENARIOS)):
            #print str(SCENARIOS[j]) + " ",
                       
            kw = kruskalwallis.kruskalwallis([aga[i][2][j],adhoc[i][2][j]], ignoreties = False)
        
            index = len(chi2_g1)-1
            while index >= 0 and kw < chi2_g1[index][1]:
                index = index - 1
            
            h0_rechazada = False
            if index >= 0:
                if kw > chi2_g1[index][1]:
                    h0_rechazada = True
            
            if h0_rechazada:
                #print "(se rechaza H0 con un p-value %f)" % (chi2_g1[index][0])
                if mean(aga[i][2][j]) > mean(adhoc[i][2][j]):
                    best_aga = best_aga + 1
                else:
                    best_adhoc = best_adhoc + 1
                
            else:
                #print "(no se puede rechazar H0)" 
                pass

        best_name = "none"
        best_count = ""
        if (best_aga > best_adhoc):
            best_name = "AGA"
            best_count = str(best_aga) + "/12"
        elif (best_aga < best_adhoc):
            best_name = "ad hoc"
            best_count = str(best_adhoc) + "/12"
        
        if best_aga == 12:
            print "& & \textbf{%.2f$\\pm$%.2f} & $%.2f\\pm%.2f$ & %s $%s$" % (aga[i][0], aga[i][1], adhoc[i][0], adhoc[i][1], best_name ,best_count)
        elif best_adhoc == 12:
            print "& & $%.2f\\pm%.2f$ & \textbf{%.2f$\\pm$%.2f} & %s $%s$" % (aga[i][0], aga[i][1], adhoc[i][0], adhoc[i][1], best_name ,best_count)
        else:
            print "& & $%.2f\\pm%.2f$ & $%.2f\\pm%.2f$ & %s $%s$" % (aga[i][0], aga[i][1], adhoc[i][0], adhoc[i][1], best_name ,best_count)

if __name__ == "__main__":
    #test()
    #test2()
    full_dimensiones()
