#encoding: utf-8
'''
Created on Feb 13, 2011

@author: santiago

'''

import nltk
import sys
import re
from PorterStemmer import PorterStemmer

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
                result.append("<<STOPWORD>>")
    
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
                result.append("<<NUMBER>>")

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
                result.append("<<SIGN>>")
        else:
            result.append(palabra)

    return result    

def remove_month_and_weekday(palabras, include_token):
    dias = [u'domingo',u'lunes',u'martes',u'miércoles',u'miercoles',u'jueves',u'viernes',u'sábado',u'sabado']
    meses = [u'enero',u'febrero',u'marzo',u'abril',u'mayo',u'junio',u'julio',u'agosto',u'setiembre',u'septiembre',u'octubre',u'noviembre',u'diciembre']

    result = []
    for palabra in palabras:
        if palabra.lower() in dias:
            if include_token:
                result.append(u'<<WEEKDAY>>')
        else:
            if palabra.lower() in meses:
                if include_token:
                    result.append(u'<<MONTH>>')
            else:
                result.append(palabra)

    return result

def word_stem(palabras):
    stemmer = PorterStemmer()

    palabras_stemmed = []
    
    for palabra in palabras:
        palabra_stemmed = stemmer.stem(palabra.lower(), 0, len(palabra)-1)
        palabras_stemmed.append(palabra_stemmed)
        
    return palabras_stemmed

def feature_word_range(featureset, oracion_left_side, oracion_right_side, epsilon_left, epsilon_right, left_tag, right_tag):
    for i in range(max(0, len(oracion_left_side)-epsilon_left), len(oracion_left_side)):
        featureset['%s(%s)' % (left_tag, oracion_left_side[i].lower())] = True
        
    for i in range(0, min(epsilon_right, len(oracion_right_side))):
        featureset['%s(%s)' % (right_tag, oracion_right_side[i].lower())] = True

    return featureset

def bigram_feature_word_range(featureset, oracion, is_left_side, tag):
    if len(oracion) >= 2:
        if not is_left_side:
            featureset['%s(%s|%s)' % (tag, \
                                      oracion[len(oracion)-2].lower(), \
                                      oracion[len(oracion)-1].lower())] = True
        else:
            featureset['%s(%s|%s)' % (tag, \
                                      oracion[0].lower(), \
                                      oracion[1].lower())] = True
        
        return featureset

def get_featureset(oracion, palabra_indice):
    palabra = oracion[palabra_indice]
    palabra_sin_tilde = sin_tilde(oracion, palabra_indice)
    
    if palabra == palabra_sin_tilde:
        label = "SIN_TILDE"
    else:
        label = "CON_TILDE"
       
    oracion_left_side = oracion[:palabra_indice]
    oracion_left_side = word_stem(oracion_left_side)
    oracion_left_side = remove_numbers(oracion_left_side, True)
    oracion_left_side = remove_month_and_weekday(oracion_left_side, True)
    #oracion_left_side = remove_stopwords(oracion_left_side, False)
    #oracion_left_side = remove_signs(oracion_left_side, True)
    oracion_left_side = oracion_sin_tilde(oracion_left_side)
    
    oracion_right_side = oracion[palabra_indice+1:]
    oracion_right_side = word_stem(oracion_right_side)
    oracion_right_side = remove_numbers(oracion_right_side, True)
    oracion_right_side = remove_month_and_weekday(oracion_right_side, True)
    #oracion_right_side = remove_stopwords(oracion_right_side, False)
    #oracion_right_side = remove_signs(oracion_right_side, True)
    oracion_right_side = oracion_sin_tilde(oracion_right_side)
        
    featureset = {}
    
    featureset['adverbio(%s)' % palabra_sin_tilde.lower()] = True
    
    #feature_word_range(featureset, oracion_left_side, oracion_right_side, 1, 0, 'contexto-1izq', 'contexto-1izq')
    #feature_word_range(featureset, oracion_left_side, oracion_right_side, 0, 1, 'contexto-1der', 'contexto-1der')
    #feature_word_range(featureset, oracion_left_side, oracion_right_side, 0, 2, 'contexto', 'contexto-1der')    
    #feature_word_range(featureset, oracion_left_side, oracion_right_side, 1, 2, 'contexto-2der1izq', 'contexto-2der1izq')
    #feature_word_range(featureset, oracion_left_side, oracion_right_side, 2, 1, 'contexto-1der2izq', 'contexto-1der2izq')   
    bigram_feature_word_range(featureset, oracion_left_side, True, 'contexto-bigramizq')
    bigram_feature_word_range(featureset, oracion_right_side, True, 'contexto-bigramder')

    capitalized = is_capitalized(palabra)
    if capitalized:
        featureset['CAPITALIZED'] = True

    question_exclamation = is_question_or_exclamation(oracion)
    if question_exclamation:
        featureset['QUESTION_EXCLAMATION'] = True

    return (featureset, label)

def oracion_decode(oracion):
    oracion_norm = []
    for palabra in oracion:
        oracion_norm.append(palabra.decode("Latin1"))
    return oracion_norm

def build_set(lista_oraciones):
    training = []
    total_con_tilde = 0
    total_sin_tilde = 0
    
    for oraciones in lista_oraciones:
        for oracion in oraciones:
            if len(oracion) > 1:
                for palabra_index in range(len(oracion)):
                    palabra_norm = oracion[palabra_index].decode("Latin1").lower()
            
                    #['cu\xe1ndo','cu\xe1nto','d\xf3nde','c\xf3mo','ad\xf3nde','qu\xe9']                
                    if palabra_norm == u'cuándo' or palabra_norm == u'cuánto' \
                    or palabra_norm == u'dónde' or palabra_norm == u'cómo' \
                    or palabra_norm == u'adónde' or palabra_norm == u'qué' \
                    or palabra_norm == u'cuando' or palabra_norm == u'cuanto' \
                    or palabra_norm == u'donde' or palabra_norm == u'como' \
                    or palabra_norm == u'adonde' or palabra_norm == u'que':
                        oracion_norm = oracion_decode(oracion)
                        features = get_featureset(oracion_norm, palabra_index)                
                        training.append(features)
                        
                        if (features[1] == "SIN_TILDE"): total_sin_tilde = total_sin_tilde + 1
                        if (features[1] == "CON_TILDE"): total_con_tilde = total_con_tilde + 1
                   
    return (training, total_sin_tilde, total_con_tilde)

def clasificar_palabra(classifier, oracion, palabra_index):
    palabra_norm = oracion[palabra_index].lower()
    
    #['cu\xe1ndo','cu\xe1nto','d\xf3nde','c\xf3mo','ad\xf3nde','qu\xe9']
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
        
def test(classifier, oraciones):
    Ok = 0
    Fail = 0
    Ok_ConTilde = 0
    Fail_ConTilde = 0

    for oracion in oraciones:
        if len(oracion) > 1:
            for palabra_index in range(len(oracion)):
                oracion_norm = oracion_decode(oracion)
                clasificacion = clasificar_palabra(classifier, oracion_norm, palabra_index)                  
                
                if clasificacion != None:
                    if (clasificacion[0] == clasificacion[1]): Ok = Ok + 1
                    else: Fail = Fail + 1
    
                    if (clasificacion[1] == 'CON_TILDE'):
                        if (clasificacion[0] == 'CON_TILDE'): Ok_ConTilde = Ok_ConTilde + 1
                        else: Fail_ConTilde = Fail_ConTilde + 1
                
    print "CANTIDAD OK: %s" % Ok
    print "CANTIDAD FAIL: %s" % Fail
    print "CANTIDAD LBASE: %s" % (Ok_ConTilde+Fail_ConTilde)
    print "CANTIDAD OK (con tilde): %s" % Ok_ConTilde
    print "CANTIDAD FAIL (con tilde): %s" % Fail_ConTilde
    
    return (Ok, Fail, Ok_ConTilde+Fail_ConTilde)
       
def run(range_index=10):
    result = []
    total = len(nltk.corpus.cess_esp.sents())
    step = total / 10
    print "TOTAL: %s CORPUS" % total
    
    pos = 0
    for pos in range(range_index):
        print "POS = %s" % pos
        test_from = pos * step
        test_to = (pos + 1) * step
        
        train_set = []
        if test_from > 0: 
            train_set.append(nltk.corpus.cess_esp.sents()[:test_from-1])
            print "[1] TRAIN FROM 0 TO %s" % (test_from-1)
        if test_to+1 < total: 
            train_set.append(nltk.corpus.cess_esp.sents()[test_to+1:])
            print "[2] TRAIN FROM %s to EOA" % (test_to+1)
        test_set = nltk.corpus.cess_esp.sents()[test_from:test_to]
        print "TEST FROM %s TO %s" % (test_from, test_to)

        print "TOTAL: %s TRAIN" % len(train_set)
        print "TOTAL: %s TEST" % len(test_set)
        
        print ">> Training..."
        train_result = build_set(train_set)
        examples = train_result[0]
        print "TOTAL: %s EXAMPLES" % len(examples)
        print "TOTAL: %s SIN_TILDE" % train_result[1]
        print "TOTAL: %s CON_TILDE" % train_result[2]
        
        classifier = nltk.NaiveBayesClassifier.train(examples)
        #classifier.show_most_informative_features(50)
        
        print ">> Testing..."
        (Ok, Fail, LBase) = test(classifier, test_set)
        result.append((pos, Ok, Fail, LBase))
        
    total_lbase = 0
    total_fail = 0
    for r in result:
        print "[POS: %s]=================================================" % (r[0])
        print "CANTIDAD OK: %s" % r[1]
        print "CANTIDAD FAIL: %s" % r[2]
        print "CANTIDAD LBASE: %s" % r[3]
        total_lbase = total_lbase + r[3]
        total_fail = total_fail + r[2]
    if total_fail < total_lbase: print ">> ENHORABUENA!!! %s < %s" % (total_fail, total_lbase)
    else: print ">> TODO MAL!!! %s > %s" % (total_fail, total_lbase)
    
if __name__ == '__main__':     
    run(1)
    