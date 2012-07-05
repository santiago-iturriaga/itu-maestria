#coding: utf-8
'''
Created on Mar 27, 2011

@author: santiago
'''

import kruskalwallis

chi2_g1 = ((0.05 ,3.84), (0.01 ,6.64), (0.001, 10.83))

def recopilar_metricas(base, instancias, scenarios, cant_iters, metricas, output_filename):
    output_file = open(output_filename, "w")

    resultados = {}

    for metrica in metricas:
        resultados[metrica] = {}
    
        for instancia in instancias:
            resultados[metrica][instancia] = []
            
            for scenario in scenarios:
                #print "****************************************************************************************"
                #print "Instancia %s" % instancia
                #print "Scenario  %s" % scenario
                #print "****************************************************************************************"
                
                #output_file.write("scenario: " + str(scenario) + ", workload: " + instancia + "\n")
                
                metrica_contenido = []
                
                for (base_path, base_prefix) in base:
                    metrica_contenido_set = []
                    
                    for i in range(cant_iters):
                        f = base_path + "/" + base_prefix + ".scenario." + str(scenario) + ".workload." + str(instancia) + "." + str(i) + "." + metrica
                        #print f
                        
                        for metrica_line in open(f, "r"):
                            valor = float(metrica_line.split(" ")[0].strip())
                            #if valor > 0:
                            metrica_contenido_set.append(valor)
                    
                    metrica_contenido.append(metrica_contenido_set)

                #print ("scenario." + str(scenario) + ".workload." + instancia)
                #print metrica_contenido[0]
                #print metrica_contenido[1]

                if len(metrica_contenido[0]) > 0 and len(metrica_contenido[1]) > 0:
                    #print "============================================"
                    #print "Metrica %s" % metrica
                    #print "============================================"
                    #print len(metrica_contenido[0])
                    #print len(metrica_contenido[1])
                    #print metrica_contenido
                    kw = kruskalwallis.kruskalwallis(metrica_contenido, ignoreties = False)
                    
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
                        output_file.write(metrica + ".scenario." + str(scenario) + ".workload." + instancia + "." + str(chi2_g1[index][0]) + "\n")
                        resultados[metrica][instancia].append(chi2_g1[index][0])
                    else:
                        #output_file.write("(no se puede rechazar H0) \n")
                        output_file.write(metrica + ".scenario." + str(scenario) + ".workload." + instancia + ".0\n")
                        resultados[metrica][instancia].append(1)
                    
                    #output_file.write("\n")
                #print "========================================="
    
    output_file.close()
    
    for metrica in metricas:
        for instancia in instancias:
            print "%s %s" % (metrica, instancia)
            m = max(resultados[metrica][instancia])
            if m == 1:
                print "N/A"
            else:
                print m

if __name__ == '__main__':
    cant_iters = 30
    
    base = []
    base.append(("metrics-aga","pals-aga"))
    base.append(("metrics-adhoc","pals-1"))
   
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
        
    metricas = []
    metricas.append("nd")
    metricas.append("hv")
    metricas.append("spread")
    metricas.append("igd")
    
    output_filename = "kw.txt"
    
    print "> Instancias:\n"
    print instancias
    print "> Métricas:\n"
    print metricas
    print "> Output:\n"
    print output_filename
    
    recopilar_metricas(base, instancias, scenario, cant_iters, metricas, output_filename)
