#coding: utf-8
'''
Created on Mar 27, 2011

@author: santiago
'''

import math

def obtener_medidas(metrica_contenido, columna):
    min = float('inf')
    max = 0
    total = 0
    nans = 0
    
    for valores in metrica_contenido:
        valor = valores[columna]
        
        if valor < min:
            min = valor
            
        if valor > max:
            max = valor
            
        if math.isnan(valor):
            nans += 1
        else:
            total += valor
        
    avg = total / (len(metrica_contenido) - nans)
    
    stdev_aux = 0
    for valores in metrica_contenido:
        valor = valores[columna]
        
        if not math.isnan(valor):
            stdev_aux += math.pow(valor - avg, 2)
        
    stdev = math.sqrt(stdev_aux / (len(metrica_contenido) - 1 - nans))
                      
    return (min, max, avg, stdev * 100 / avg)

def recopilar_metricas(instancias, metricas, output_filename):
    output_file = open(output_filename, "w")
    
    # Escribo los cabezales del CSV.
    cabezal = "instancia,"
    for metrica in metricas:
        (nombre_metrica, archivo_metrica, columnas_metrica) = metrica
        
        if (columnas_metrica > 1):
            for col in range(columnas_metrica):
                cabezal += nombre_metrica + "_" + str(col) + "-min,"
                cabezal += nombre_metrica + "_" + str(col) + "-max,"
                cabezal += nombre_metrica + "_" + str(col) + "-avg,"
                cabezal += nombre_metrica + "_" + str(col) + "-stdev,"
        else:
            cabezal += nombre_metrica + "-min,"
            cabezal += nombre_metrica + "-max,"
            cabezal += nombre_metrica + "-avg,"
            cabezal += nombre_metrica + "-stdev,"
    
    output_file.write(cabezal + "\n")

    for instancia in instancias:
        (nombre_instancia, ubicacion_instancia) = instancia
        
        output_file.write(nombre_instancia + ",")
        
        for metrica in metricas:
            (nombre_metrica, archivo_metrica, columnas_metrica) = metrica
            
            metrica_contenido = []
            for metrica_line in open(ubicacion_instancia + "/" + archivo_metrica, "r"):
                metrica_line_aux = metrica_line.split(",")
                
                if len(metrica_line_aux) == columnas_metrica:
                    metrica_contenido_aux = []
                    
                    for col in range(columnas_metrica):
                        valor = float(metrica_line_aux[col].strip())
                        metrica_contenido_aux.append(valor)
                        
                    metrica_contenido.append(metrica_contenido_aux)
                else:
                    exit(-1)
            
            for col in range(columnas_metrica):
                (metrica_min, metrica_max, metrica_avg, metrica_stdev) = obtener_medidas(metrica_contenido, col)
                output_file.write(str(metrica_min) + "," + str(metrica_max) + "," + str(metrica_avg) + "," + str(metrica_stdev) + ",")  
            
        output_file.write("\n")
    
    output_file.close()

if __name__ == '__main__':
    instancias = []
    instancias.append(("u_c_hihi.0", "./u_c_hihi.0"))
    instancias.append(("u_c_hilo.0", "./u_c_hilo.0"))
    instancias.append(("u_c_lohi.0", "./u_c_lohi.0"))
    instancias.append(("u_c_lolo.0", "./u_c_lolo.0"))
    instancias.append(("u_i_hihi.0", "./u_i_hihi.0"))
    instancias.append(("u_i_hilo.0", "./u_i_hilo.0"))
    instancias.append(("u_i_lohi.0", "./u_i_lohi.0"))
    instancias.append(("u_i_lolo.0", "./u_i_lolo.0"))
    instancias.append(("u_s_hihi.0", "./u_s_hihi.0"))
    instancias.append(("u_s_hilo.0", "./u_s_hilo.0"))
    instancias.append(("u_s_lohi.0", "./u_s_lohi.0"))
    instancias.append(("u_s_lolo.0", "./u_s_lolo.0"))

    #instancias.append(("A.u_c_hihi", "./A.u_c_hihi"))
#    instancias.append(("A.u_c_hilo", "./A.u_c_hilo"))
#    instancias.append(("A.u_c_lohi", "./A.u_c_lohi"))
#    instancias.append(("A.u_c_lolo", "./A.u_c_lolo"))
#    instancias.append(("A.u_i_hihi", "./A.u_i_hihi"))
#    instancias.append(("A.u_i_hilo", "./A.u_i_hilo"))
#    instancias.append(("A.u_i_lohi", "./A.u_i_lohi"))
#    instancias.append(("A.u_i_lolo", "./A.u_i_lolo"))
#    instancias.append(("A.u_s_hihi", "./A.u_s_hihi"))
#    instancias.append(("A.u_s_hilo", "./A.u_s_hilo"))
#    instancias.append(("A.u_s_lohi", "./A.u_s_lohi"))
#    instancias.append(("A.u_s_lolo", "./A.u_s_lolo"))

#    instancias.append(("B.u_c_hihi", "./B.u_c_hihi"))
#    instancias.append(("B.u_c_hilo", "./B.u_c_hilo"))
#    instancias.append(("B.u_c_lohi", "./B.u_c_lohi"))
#    instancias.append(("B.u_c_lolo", "./B.u_c_lolo"))
#    instancias.append(("B.u_i_hihi", "./B.u_i_hihi"))
#    instancias.append(("B.u_i_hilo", "./B.u_i_hilo"))
#    instancias.append(("B.u_i_lohi", "./B.u_i_lohi"))
#    instancias.append(("B.u_i_lolo", "./B.u_i_lolo"))
#    instancias.append(("B.u_s_hihi", "./B.u_s_hihi"))
#    instancias.append(("B.u_s_hilo", "./B.u_s_hilo"))
#    instancias.append(("B.u_s_lohi", "./B.u_s_lohi"))
#    instancias.append(("B.u_s_lolo", "./B.u_s_lolo"))

    metricas = []
    metricas.append(("Spread", "reporte_spread.txt", 1))
    metricas.append(("Spacing", "reporte_spacing.txt", 1))
    metricas.append(("GD", "reporte_gd.txt", 2))
    
    output_filename = "result.csv"
    
    print "> Instancias:\n"
    print instancias
    print "> Métricas:\n"
    print metricas
    print "> Output:\n"
    print output_filename
    
    recopilar_metricas(instancias, metricas, output_filename)
