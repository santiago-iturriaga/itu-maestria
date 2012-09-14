#coding: utf-8
'''
Created on Mar 27, 2011

@author: santiago
'''

import kruskalwallis

chi2_g1 = ((0.05 ,3.84), (0.01 ,6.64), (0.001, 10.83))

def recopilar_metricas(base_paths, instancias, metricas, output_filename):
    output_file = open(output_filename, "w")
    
    for instancia in instancias:
        print "****************************************************************************************"
        print "Instancia %s" % instancia
        print "****************************************************************************************"
        
        output_file.write(instancia + "\n")
        
        for (metrica,col) in metricas:
            metrica_contenido = []
            
            for base_path in base_paths:
                metrica_contenido_set = []
                
                for metrica_line in open(base_path + instancia + "/" + metrica, "r"):
                    valor = float(metrica_line.split(",")[col].strip())
                    metrica_contenido_set.append(valor)
                
                metrica_contenido.append(metrica_contenido_set)

            print "============================================"
            print "Metrica %s" % metrica
            print "============================================"
            print metrica_contenido
            kw = kruskalwallis.kruskalwallis(metrica_contenido, ignoreties = False)
            
            index = len(chi2_g1)-1
            while index >= 0 and kw < chi2_g1[index][1]:
                index = index - 1
            
            output_file.write("Metrica %s => resultado Kruskal-Wallis = %f " % (metrica, kw))
            
            h0_rechazada = False
            if index >= 0:
                if kw > chi2_g1[index][1]:
                    h0_rechazada = True
            
            if h0_rechazada:
                output_file.write("(se rechaza H0 con un p-value %f) \n" % (chi2_g1[index][0]))
            else:
                output_file.write("(no se puede rechazar H0) \n")
            
        output_file.write("\n")
    
    output_file.close()

if __name__ == '__main__':
    base_paths = []
    base_paths.append("/home/siturria/AE/Metricas/mo/MOCHC_Braun/")
    base_paths.append("/home/siturria/AE/Metricas/barca/1/")
    
    instancias = []
    instancias.append("u_c_hihi.0")
    instancias.append("u_c_hilo.0")
    instancias.append("u_c_lohi.0")
    instancias.append("u_c_lolo.0")
    instancias.append("u_i_hihi.0")
    instancias.append("u_i_hilo.0")
    instancias.append("u_i_lohi.0")
    instancias.append("u_i_lolo.0")
    instancias.append("u_s_hihi.0")
    instancias.append("u_s_hilo.0")
    instancias.append("u_s_lohi.0")
    instancias.append("u_s_lolo.0")
        
    metricas = []
    metricas.append(("reporte_spread.txt",0))
    metricas.append(("reporte_spacing.txt",0))
    metricas.append(("reporte_gd.txt",1))
    
    output_filename = "kw.txt"
    
    print "> Instancias:\n"
    print instancias
    print "> Métricas:\n"
    print metricas
    print "> Output:\n"
    print output_filename
    
    recopilar_metricas(base_paths, instancias, metricas, output_filename)