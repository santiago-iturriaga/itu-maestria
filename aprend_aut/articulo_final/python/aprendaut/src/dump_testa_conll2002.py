#encoding: utf-8
'''
Created on Feb 15, 2011

@author: santiago
'''

import nltk
import re

def sin_tilde(palabra):
    if palabra == u'cuándo':
        return u'cuando'
    elif palabra == u'cuánto':
        return u'cuanto'
    elif palabra == u'dónde':
        return u'donde'
    elif palabra == u'cómo':
        return u'como'
    elif palabra == u'adónde':
        return u'adonde'
    elif palabra == u'qué':
        return u'que'
    else:
        return palabra

def is_capitalized(palabra):
    if palabra[0].upper() == palabra[0]: return True
    else: return False
    
def is_stopword(stop_words, palabra):   
    if palabra.lower() in stop_words:
        return True
    else:
        return False

def is_sign(palabra):
    if palabra == u"." or palabra == u"," \
    or palabra == u";" or palabra == u":" \
    or palabra == u"-" or palabra == u"\"" \
    or palabra == u"(" or palabra == u")" \
    or palabra == u"*" or palabra == u"$" \
    or palabra == u"[" or palabra == u"]" \
    or palabra == u"{" or palabra == u"}" \
    or palabra == u"+" or palabra == u"&" \
    or palabra == u"<" or palabra == u">":
        return True
    else:
        return False

def is_question(palabra):
    if palabra == u"¿" or palabra == u"?":
        return True
    else:
        return False
    
def is_exclamation(palabra):
    if palabra == u"¡" or palabra == u"!": 
        return True
    else:
        return False


def is_number(palabra):
    if re.match("(.*)[0-9](.*)",palabra) == None:
        return False
    else:
        return True

if __name__ == '__main__':
    stop_words = set(nltk.corpus.stopwords.words('spanish'))
   
    classify_file = open("esp.testa-classify.txt", "w")
    classify_file_check = open("esp.testa-classify-check.txt", "w")
    classify_file_check_full = open("esp.testa-classify-check-full.txt", "w")
    
    for oracion in nltk.corpus.conll2002.sents('esp.testa'):
        if len(oracion) > 1:
            for palabra in oracion:
                if len(palabra) > 0:
                    palabra_norm = palabra.lower()
                       
                    if palabra_norm == u'cuándo' or palabra_norm == u'cuánto' \
                    or palabra_norm == u'dónde' or palabra_norm == u'cómo' \
                    or palabra_norm == u'adónde' or palabra_norm == u'qué' \
                    or palabra_norm == u'cuando' or palabra_norm == u'cuanto' \
                    or palabra_norm == u'donde' or palabra_norm == u'como' \
                    or palabra_norm == u'adonde' or palabra_norm == u'que':
                        palabra_sin_tilde = sin_tilde(palabra_norm)
                        
                        if palabra_sin_tilde == palabra_norm:
                            label = u"SIN_TILDE"
                        else:
                            label = u"CON_TILDE"
        
                        palabra_procesada = palabra_sin_tilde
                    else:
                        label = u"O"
                        palabra_procesada = palabra_norm
                       
                    features = ""
        
                    if is_sign(palabra):
                        features = features + " SIGN"
                    if is_question(palabra):
                        features = features + " QUESTION"
                    if is_exclamation(palabra):
                        features = features + " EXCLAMATION"
                    else:           
                        if is_number(palabra):
                            features = features + " NUMBER"
                        else:
                            if is_capitalized(palabra):
                                features = features + " CAPITALIZED"   
            
                            if is_stopword(stop_words, palabra):
                                features = features + " STOPWORD"
                       
                    if label == u"O":
                        classify_file.write("%s %s %s\n" % (palabra_procesada.encode('utf8'), features.strip(), label.encode('utf8')))
                    else:
                        classify_file.write("%s %s\n" % (palabra_procesada.encode('utf8'), features.strip()))
                        
                    classify_file_check.write("%s\n" % (label.encode('utf8')))
                    classify_file_check_full.write("%s %s\n" % (palabra_procesada.encode('utf8'), label.encode('utf8')))            
            
            classify_file.write("\n")
            classify_file_check.write("\n")
            classify_file_check_full.write("\n")
            
    classify_file.close()
    classify_file_check.close()
    classify_file_check_full.close()