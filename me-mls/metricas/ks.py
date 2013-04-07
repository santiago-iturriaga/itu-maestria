#coding: utf-8
'''
Created on Mar 27, 2011

@author: santiago
'''

import numpy as np
import scipy.stats as stats

def recopilar_metricas(base, instancias, scenarios, cant_iters):
    resultados = {}

    for instancia in instancias:
        resultados[instancia] = ([],[])
        
        for scenario in scenarios:
            metrica_contenido_mk = []
            metrica_contenido_nrg = []
                          
            for (base_path, base_prefix) in base:
                metrica_contenido_set_mk = []
                metrica_contenido_set_nrg = []
                
                for i in range(cant_iters):
                    f = base_path  + "/scenario." + str(scenario) + ".workload." + str(instancia) + "." + str(i) + "/" + base_prefix + ".scenario." + str(scenario) + ".workload." + str(instancia) + ".metrics"
                    #print f

                    min_mk = 0.0
                    min_nrg = 0.0
                    
                    for metrica_line in open(f, "r"):
                        valor_mk = float(metrica_line.split(" ")[0].strip())
                        valor_nrg = float(metrica_line.split(" ")[1].strip())

                        if min_mk == 0.0: min_mk = valor_mk
                        elif min_mk > valor_mk: min_mk = valor_mk

                        if min_nrg == 0.0: min_nrg = valor_nrg
                        elif min_nrg > valor_nrg: min_nrg = valor_nrg
                        
                    metrica_contenido_set_mk.append(min_mk)
                    metrica_contenido_set_nrg.append(min_nrg)
                
                metrica_contenido_mk.append(metrica_contenido_set_mk)
                metrica_contenido_nrg.append(metrica_contenido_set_nrg)

            if len(metrica_contenido_mk[0]) > 0 and len(metrica_contenido_nrg[1]) > 0 and len(metrica_contenido_mk[0]) > 0 and len(metrica_contenido_nrg[1]) > 0:                          
                data=np.array(metrica_contenido_mk[0])
                normed_data=(data-data.mean())/data.std()
                (d1,p1)=stats.kstest(normed_data,'norm')
                
                data=np.array(metrica_contenido_mk[1])
                normed_data=(data-data.mean())/data.std()
                (d2,p2)=stats.kstest(normed_data,'norm')
                
                data=np.array(metrica_contenido_nrg[0])
                normed_data=(data-data.mean())/data.std()
                (d3,p3)=stats.kstest(normed_data,'norm')
                
                data=np.array(metrica_contenido_nrg[1])
                normed_data=(data-data.mean())/data.std()
                (d4,p4)=stats.kstest(normed_data,'norm')
                
                print "===== Scenario: " + str(scenario) + ", Workload: " + instancia
                if p1 > 0.05:
                    print "Makespan AGA   : %s" % (p1)
                if p2 > 0.05:
                    print "Makespan Ad hoc: %s" % (p2)
                if p3 > 0.05:
                    print "Energy AGA     : %s" % (p3)
                if p4 > 0.05:
                    print "Energy AGA     : %s" % (p4)

    
if __name__ == '__main__':
    cant_iters = 30
    
    base = []
    base.append(("../512x16.24_10s","pals-aga"))
    base.append(("../512x16.24.adhoc","pals-1"))
    print base
   
    scenario = []
    scenario.append(0)
    scenario.append(3)
    scenario.append(6)
    scenario.append(9)
    scenario.append(10)
    scenario.append(11)
    scenario.append(13)
    scenario.append(14)
    scenario.append(16)
    scenario.append(17)
    scenario.append(19)
    
    instancias = []
    instancias.append("A.u_c_hihi")
    instancias.append("A.u_c_hilo")
    instancias.append("A.u_c_lohi")
    instancias.append("A.u_c_lolo")
    instancias.append("A.u_i_hihi")
    instancias.append("A.u_i_hilo")
    instancias.append("A.u_i_lohi")
    instancias.append("A.u_i_lolo")
    instancias.append("A.u_s_hihi")
    instancias.append("A.u_s_hilo")
    instancias.append("A.u_s_lohi")
    instancias.append("A.u_s_lolo")
    instancias.append("B.u_c_hihi")
    instancias.append("B.u_c_hilo")
    instancias.append("B.u_c_lohi")
    instancias.append("B.u_c_lolo")
    instancias.append("B.u_i_hihi")
    instancias.append("B.u_i_hilo")
    instancias.append("B.u_i_lohi")
    instancias.append("B.u_i_lolo")
    instancias.append("B.u_s_hihi")
    instancias.append("B.u_s_hilo")
    instancias.append("B.u_s_lohi")
    instancias.append("B.u_s_lolo")  
           
    print "> Instancias:\n"
    print instancias
    
    recopilar_metricas(base, instancias, scenario, cant_iters)
