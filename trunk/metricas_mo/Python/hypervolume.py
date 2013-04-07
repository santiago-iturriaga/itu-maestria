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

def kw(hv_alg_1, hv_alg_2):
    kw = kruskalwallis.kruskalwallis(hv_alg_1,hv_alg_2, ignoreties = False)

    index = len(chi2_g1)-1
    while index >= 0 and kw < chi2_g1[index][1]:
        index = index - 1
    
    h0_rechazada = False
    if index >= 0:
        if kw > chi2_g1[index][1]:
            h0_rechazada = True
    
    return h0_rechazada   
    
if __name__ == "__main__":
    pass
