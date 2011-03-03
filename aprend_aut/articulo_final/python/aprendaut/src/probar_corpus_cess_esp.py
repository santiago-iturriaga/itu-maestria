#encoding: utf-8
'''
Created on Feb 12, 2011

@author: santiago
'''

import nltk

if __name__ == '__main__':
    cantidad = 0
    cantidad_no_adv = 0
    
    for oracion in nltk.corpus.cess_esp.sents():
        contenido = ""
        contiene_adv_interr = False
        contiene_no_adv_interr = False
        
        for palabra in oracion:
            contenido = contenido + " " + palabra
            palabra_norm = palabra.lower()
            
            # cuándo
            # cuánto
            # dónde
            # cómo
            # adónde
            
            if palabra_norm == 'cu\xe1ndo':
                contiene_adv_interr = True
            elif palabra_norm == 'cu\xe1nto':
                contiene_adv_interr = True
            elif palabra_norm == 'd\xf3nde':
                contiene_adv_interr = True
            elif palabra_norm == 'c\xf3mo':
                contiene_adv_interr = True
            elif palabra_norm == 'ad\xf3nde':
                contiene_adv_interr = True

            if palabra_norm == 'cuando':
                contiene_no_adv_interr = True
            elif palabra_norm == 'cuanto':
                contiene_no_adv_interr = True
            elif palabra_norm == 'donde':
                contiene_no_adv_interr = True
            elif palabra_norm == 'como':
                contiene_no_adv_interr = True
            elif palabra_norm == 'adonde':
                contiene_no_adv_interr = True
        
        if contiene_adv_interr:
            cantidad = cantidad + 1
            print " - " + contenido
            print "-------------------------------------------------------------------"
            
        if contiene_no_adv_interr:
            cantidad_no_adv = cantidad_no_adv + 1
            
    print "TOTAL: %s " % cantidad
    print "TOTAL NO ADV: %s " % cantidad_no_adv