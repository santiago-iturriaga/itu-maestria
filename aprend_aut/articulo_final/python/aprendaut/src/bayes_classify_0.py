#encoding: utf-8
'''
Created on Feb 13, 2011

@author: santiago

CANTIDAD OK: 11779
CANTIDAD FAIL: 285
CANTIDAD OK (con tilde): 50
CANTIDAD FAIL (con tilde): 25

'''

import nltk
import sys
import re

def oracion_sin_tilde(oracion):
    result = []
    for palabra in oracion:
        result.append(palabra_sin_tilde(palabra))        
    return result

def palabra_sin_tilde(palabra):
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
    
    return palabra

def sin_tilde(oracion, indice):
    return palabra_sin_tilde(oracion[indice])

def bag_of_words(palabras):
    result = {}
    
    for palabra in palabras:
        if not result.has_key(palabra.lower()):
            result[palabra] = True

    return result

def lower_case(palabras):
    return [palabra.lower() for palabra in palabras]

def remove_stopwords(palabras, include_token):
    result = []
    stop_words = set(nltk.corpus.stopwords.words('spanish'))
    
    for palabra in palabras:
        if palabra.lower() not in stop_words:
            result.append(palabra)
        else:
            if include_token:
                result.append("STOPWORD")
    
    return result

def is_capitalized(palabra):
    if palabra[0].upper() == palabra[0]: return True
    else: return False

def remove_numbers(palabras, include_token):
    result = []
    
    for palabra in palabras:
        if re.match("(.*)[0-9](.*)",palabra) == None:
            result.append(palabra)
        else:
            if include_token:
                result.append("NUMBER")

    return result

def is_question_or_exclamation(palabras):
    for palabra in palabras:
        if re.match("(.*)[¿|?|¡|!](.*)",palabra) != None:
            return True

    return False

def remove_signs(palabras, include_token):
    result = []

    for palabra in palabras:
        if palabra == u"." or palabra == u"," \
        or palabra == u";" or palabra == u":" \
        or palabra == u"-" or palabra == u"\"" \
        or palabra == u"(" or palabra == u")" \
        or palabra == u"*" or palabra == u"$" \
        or palabra == u"[" or palabra == u"]" \
        or palabra == u"{" or palabra == u"}" \
        or palabra == u"¡" or palabra == u"!" \
        or palabra == u"¿" or palabra == u"?" \
        or palabra == u"+" or palabra == u"&" \
        or palabra == u"<" or palabra == u">":
            if include_token:
                result.append("SIGN")
        else:
            result.append(palabra)

    return result    

def get_featureset(oracion, palabra_indice):
    palabra = oracion[palabra_indice]
    palabra_sin_tilde = sin_tilde(oracion, palabra_indice)
    
    if palabra == palabra_sin_tilde:
        label = u"SIN_TILDE"
    else:
        label = u"CON_TILDE"
    
    epsilon_left = 1
    epsilon_right = 0
    
    oracion_left_side = oracion[:palabra_indice]
    oracion_left_side = remove_numbers(oracion_left_side, False)
    oracion_left_side = remove_stopwords(oracion_left_side, False)
    #oracion_left_side = remove_signs(oracion_left_side, False)
    oracion_left_side = oracion_sin_tilde(oracion_left_side)
    
    oracion_right_side = oracion[palabra_indice+1:]
    oracion_right_side = remove_numbers(oracion_right_side, False)
    oracion_right_side = remove_stopwords(oracion_right_side, False)
    #oracion_right_side = remove_signs(oracion_right_side, False)
    oracion_right_side = oracion_sin_tilde(oracion_right_side)
        
    featureset = {}
    
    for i in range(max(0, len(oracion_left_side)-epsilon_left), len(oracion_left_side)):
        featureset['contexto-izq(%s)' % oracion_left_side[i].lower()] = True
        
    featureset['contexto(%s)' % palabra_sin_tilde.lower()] = True
        
    for i in range(0, min(epsilon_right, len(oracion_right_side))):
        featureset['contexto-der(%s)' % oracion_right_side[i].lower()] = True

    capitalized = is_capitalized(palabra)
    if capitalized:
        featureset['CAPITALIZED'] = True

    question_exclamation = is_question_or_exclamation(oracion)
    if question_exclamation:
        featureset['QUESTION_EXCLAMATION'] = True

    return (featureset, label)

def build_set():
    training = []
    
    for oracion in nltk.corpus.conll2002.sents('esp.train'):
        for palabra_index in range(len(oracion)):
            palabra_norm = oracion[palabra_index].lower()
                    
            if palabra_norm == u'cuándo' or palabra_norm == u'cuánto' \
            or palabra_norm == u'dónde' or palabra_norm == u'cómo' \
            or palabra_norm == u'adónde' or palabra_norm == u'qué' \
            or palabra_norm == u'cuando' or palabra_norm == u'cuanto' \
            or palabra_norm == u'donde' or palabra_norm == u'como' \
            or palabra_norm == u'adonde' or palabra_norm == u'que':
                features = get_featureset(oracion, palabra_index)                
                training.append(features)
                   
    return training

def clasificar_palabra(classifier, oracion, palabra_index):
    palabra_norm = oracion[palabra_index].lower()
    
    if palabra_norm == u'cuándo' or palabra_norm == u'cuánto' \
    or palabra_norm == u'dónde' or palabra_norm == u'cómo' \
    or palabra_norm == u'adónde' or palabra_norm == u'qué' \
    or palabra_norm == u'cuando' or palabra_norm == u'cuanto' \
    or palabra_norm == u'donde' or palabra_norm == u'como' \
    or palabra_norm == u'adonde' or palabra_norm == u'que':
        (features, label) = get_featureset(oracion, palabra_index)
        result = classifier.classify(features)        
        return (result, label, features)
    
    return None
    
def test(classifier):
    Ok = 0
    Fail = 0
    Ok_ConTilde = 0
    Fail_ConTilde = 0

    Total_Ok = 0
    Total_Fail = 0
    Total_Ok_ConTilde = 0
    Total_Fail_ConTilde = 0
    
    for oracion in nltk.corpus.conll2002.sents('esp.testa'):
        for palabra_index in range(len(oracion)):
            clasificacion = clasificar_palabra(classifier, oracion, palabra_index)                  
            
            if clasificacion != None:
                if (clasificacion[0] == clasificacion[1]): Ok = Ok + 1
                else: Fail = Fail + 1

                if (clasificacion[1] == u'CON_TILDE'):
                    if (clasificacion[0] == u'CON_TILDE'): Ok_ConTilde = Ok_ConTilde + 1
                    else: Fail_ConTilde = Fail_ConTilde + 1
                
    print "[esp.testa] CANTIDAD OK: % s" % Ok
    print "[esp.testa] CANTIDAD FAIL: % s" % Fail
    print "[esp.testa] CANTIDAD OK (con tilde): % s" % Ok_ConTilde
    print "[esp.testa] CANTIDAD FAIL (con tilde): % s" % Fail_ConTilde
    print "\n"
       
    Total_Ok = Total_Ok + Ok
    Total_Fail = Total_Fail + Fail
    Total_Ok_ConTilde = Total_Ok_ConTilde + Ok_ConTilde
    Total_Fail_ConTilde = Total_Fail_ConTilde + Fail_ConTilde
             
    # ======================================================================
                
    Ok = 0
    Fail = 0
    Ok_ConTilde = 0
    Fail_ConTilde = 0
                
    for oracion in nltk.corpus.conll2002.sents('esp.testb'):
        for palabra_index in range(len(oracion)):
            clasificacion = clasificar_palabra(classifier, oracion, palabra_index)                  
            
            if clasificacion != None:
                if (clasificacion[0] == clasificacion[1]): Ok = Ok + 1
                else: Fail = Fail + 1

                if (clasificacion[1] == u'CON_TILDE'):
                    if (clasificacion[0] == u'CON_TILDE'): Ok_ConTilde = Ok_ConTilde + 1
                    else: Fail_ConTilde = Fail_ConTilde + 1
    
    print "[esp.testb] CANTIDAD OK: % s" % Ok
    print "[esp.testb] CANTIDAD FAIL: % s" % Fail

    print "[esp.testb] CANTIDAD OK (con tilde): % s" % Ok_ConTilde
    print "[esp.testb] CANTIDAD FAIL (con tilde): % s" % Fail_ConTilde
    print "\n"
    
    Total_Ok = Total_Ok + Ok
    Total_Fail = Total_Fail + Fail
    Total_Ok_ConTilde = Total_Ok_ConTilde + Ok_ConTilde
    Total_Fail_ConTilde = Total_Fail_ConTilde + Fail_ConTilde

    # ======================================================================
                
    Ok = 0
    Fail = 0
    Ok_ConTilde = 0
    Fail_ConTilde = 0
                
    for oracion in nltk.corpus.conll2002.sents('esp.train'):
        for palabra_index in range(len(oracion)):
            clasificacion = clasificar_palabra(classifier, oracion, palabra_index)                  
            
            if clasificacion != None:
                if (clasificacion[0] == clasificacion[1]): Ok = Ok + 1
                else: Fail = Fail + 1

                if (clasificacion[1] == u'CON_TILDE'):
                    if (clasificacion[0] == u'CON_TILDE'): Ok_ConTilde = Ok_ConTilde + 1
                    else: Fail_ConTilde = Fail_ConTilde + 1
    
    print "[esp.train] CANTIDAD OK: % s" % Ok
    print "[esp.train] CANTIDAD FAIL: % s" % Fail

    print "[esp.train] CANTIDAD OK (con tilde): % s" % Ok_ConTilde
    print "[esp.train] CANTIDAD FAIL (con tilde): % s" % Fail_ConTilde
    print "\n"
    
    Total_Ok = Total_Ok + Ok
    Total_Fail = Total_Fail + Fail
    Total_Ok_ConTilde = Total_Ok_ConTilde + Ok_ConTilde
    Total_Fail_ConTilde = Total_Fail_ConTilde + Fail_ConTilde
    
    # ======================================================================
    
    print "CANTIDAD OK: % s" % Total_Ok
    print "CANTIDAD FAIL: % s" % Total_Fail

    print "CANTIDAD OK (con tilde): % s" % Total_Ok_ConTilde
    print "CANTIDAD FAIL (con tilde): % s" % Total_Fail_ConTilde
    
if __name__ == '__main__':
    examples = build_set()  
    print "TOTAL: %s EXAMPLES" % len(examples)
    
    classifier = nltk.NaiveBayesClassifier.train(examples)
    #classifier.show_most_informative_features(50)
    
    test(classifier)
  