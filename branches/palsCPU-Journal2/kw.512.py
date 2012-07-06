#coding: utf-8
'''
Created on Mar 27, 2011

@author: santiago
'''

import kruskalwallis

chi2_g1 = ((0.05 ,3.84), (0.01 ,6.64), (0.001, 10.83))

def recopilar_metricas(base, instancias, scenarios, cant_iters, output_filename):
    output_file = open(output_filename, "w")

    resultados = {}

    for instancia in instancias:
        resultados[instancia] = ([],[])
        
        for scenario in scenarios:
            #print "****************************************************************************************"
            #print "Instancia %s" % instancia
            #print "Scenario  %s" % scenario
            #print "****************************************************************************************"
                          
            #output_file.write("scenario: " + str(scenario) + ", workload: " + instancia + "\n")

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

            #print ("scenario." + str(scenario) + ".workload." + instancia)
            #print metrica_contenido[0]
            #print metrica_contenido[1]

            if len(metrica_contenido_mk[0]) > 0 and len(metrica_contenido_nrg[1]) > 0 and len(metrica_contenido_mk[0]) > 0 and len(metrica_contenido_nrg[1]) > 0:
                #print "============================================"
                #print "Metrica %s" % metrica
                #print "============================================"
                #print len(metrica_contenido[0])
                #print len(metrica_contenido[1])
                #print metrica_contenido
                kw = kruskalwallis.kruskalwallis(metrica_contenido_mk, ignoreties = False)
                
                index = len(chi2_g1)-1
                while index >= 0 and kw < chi2_g1[index][1]:
                    index = index - 1
                
                #output_file.write("Metrica %s => resultado Kruskal-Wallis = %f " % (metrica, kw))
                
                h0_rechazada = False
                if index >= 0:
                    if kw > chi2_g1[index][1]:
                        h0_rechazada = True
                
                if h0_rechazada:
                    #output_file.write("(se rechaza H0 con un p-value %f) \n" % (chi2_g1[index][0]))
                    output_file.write("scenario." + str(scenario) + ".workload." + instancia + "." + str(chi2_g1[index][0]) + "\n")
                    resultados[instancia][0].append(chi2_g1[index][0])
                else:
                    #output_file.write("(no se puede rechazar H0) \n")
                    output_file.write("scenario." + str(scenario) + ".workload." + instancia + ".0\n")
                    resultados[instancia][0].append(1)
                    
                # ================================================
                
                kw = kruskalwallis.kruskalwallis(metrica_contenido_nrg, ignoreties = False)
                
                index = len(chi2_g1)-1
                while index >= 0 and kw < chi2_g1[index][1]:
                    index = index - 1
                
                #output_file.write("Metrica %s => resultado Kruskal-Wallis = %f " % (metrica, kw))
                
                h0_rechazada = False
                if index >= 0:
                    if kw > chi2_g1[index][1]:
                        h0_rechazada = True
                
                if h0_rechazada:
                    #output_file.write("(se rechaza H0 con un p-value %f) \n" % (chi2_g1[index][0]))
                    output_file.write("scenario." + str(scenario) + ".workload." + instancia + "." + str(chi2_g1[index][0]) + "\n")
                    resultados[instancia][1].append(chi2_g1[index][0])
                else:
                    #output_file.write("(no se puede rechazar H0) \n")
                    output_file.write("scenario." + str(scenario) + ".workload." + instancia + ".0\n")
                    resultados[instancia][1].append(1)
                
                #output_file.write("\n")
            #print "========================================="
    
    output_file.close()
    
    for instancia in instancias:
        print "%s" % (instancia)
        m = max(resultados[instancia][0])
        if m == 1:
            print "N/A",
        else:
            print m,

        m = max(resultados[instancia][1])
        if m == 1:
            print " N/A"
        else:
            print " %s" % m


if __name__ == '__main__':
    cant_iters = 30
    
    base = []
    base.append(("512x16.24_10s","pals-aga"))
    base.append(("512x16.24.adhoc","pals-1"))
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
    instancias.append("B.u_c_hihi")
    instancias.append("B.u_c_hilo")
    instancias.append("B.u_c_lohi")
    instancias.append("B.u_c_lolo")
           
    output_filename = "kw.txt"
    
    print "> Instancias:\n"
    print instancias
    print "> Output:\n"
    print output_filename
    
    recopilar_metricas(base, instancias, scenario, cant_iters, output_filename)
