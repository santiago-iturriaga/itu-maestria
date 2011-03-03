#encoding: utf-8
'''
Created on Feb 15, 2011

@author: santiago
'''

import nltk

def get_previous_gram(oracion, indice):
    if indice == 0:
        return '.'
    else:
        return oracion[indice-1].lower()

def get_next_gram(oracion, indice):
    if indice == len(oracion)-1:
        return '.'
    else:
        return oracion[indice+1].lower()

def sin_tilde(oracion, indice):
    if oracion[indice] == u'cuándo':
        return u'cuando'
    elif oracion[indice] == u'cuánto':
        return u'cuanto'
    elif oracion[indice] == u'dónde':
        return u'donde'
    elif oracion[indice] == u'cómo':
        return u'como'
    elif oracion[indice] == u'adónde':
        return u'adonde'
    elif oracion[indice] == u'qué':
        return u'que'
    
    return oracion[indice]

def get_features_mejor_bayes(oracion, indice):
    featureset = ""
    
    for i in range(max(0, indice-1), indice):
        featureset = featureset + " contexto-izq(%s)" % oracion[i].lower()
        
    featureset = featureset + " contexto(%s)" % oracion[indice].lower()
                
    return featureset

if __name__ == '__main__':
    corpus_file = open("corpus.txt", "w")
    corpus_file_orig = open("corpus-orig.txt", "w")
    
    for oracion_index in range(1000):
        oracion = nltk.corpus.conll2002.sents()[oracion_index]
        
        for palabra in oracion:
            palabra_norm = palabra.lower()

            if palabra_norm == u'cuándo' or palabra_norm == u'cuánto' \
                or palabra_norm == u'dónde' or palabra_norm == u'cómo' \
                or palabra_norm == u'adónde' or palabra_norm == u'qué' \
                or palabra_norm == u'cuando' or palabra_norm == u'cuanto' \
                or palabra_norm == u'donde' or palabra_norm == u'como' \
                or palabra_norm == u'adonde' or palabra_norm == u'que':
            
                corpus_file_orig.write("%s\n" % (palabra_norm.encode('utf8')))
                
                if palabra_norm == u'cuándo' or palabra_norm == u'cuánto' \
                    or palabra_norm == u'dónde' or palabra_norm == u'cómo' \
                    or palabra_norm == u'adónde' or palabra_norm == u'qué':
                    
                    corpus_file.write("%s CON_TILDE\n" % (sin_tilde(palabra_norm).encode('utf8')))
                    
                elif palabra_norm == u'cuando' or palabra_norm == u'cuanto' \
                    or palabra_norm == u'donde' or palabra_norm == u'como' \
                    or palabra_norm == u'adonde' or palabra_norm == u'que':
                    
                    corpus_file.write("%s SIN_TILDE\n" % (palabra_norm.encode('utf8')))
    
                else:
                    corpus_file.write("%s O\n" % (palabra_norm.encode('utf8')))
            
    corpus_file.close()
    corpus_file_orig.close()