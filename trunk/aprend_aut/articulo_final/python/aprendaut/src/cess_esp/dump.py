#encoding: utf-8
'''
Created on Feb 15, 2011

@author: santiago
'''

import nltk

def get_label(palabra):
    if palabra == u'cuándo' \
    or palabra == u'cuánto' \
    or palabra == u'dónde' \
    or palabra == u'cómo' \
    or palabra == u'adónde' \
    or palabra == u'qué':
        return u'CON_TILDE'
    elif palabra == u'cuando' \
    or palabra == u'cuanto' \
    or palabra == u'donde' \
    or palabra == u'como' \
    or palabra == u'adonde' \
    or palabra == u'que':
        return u'SIN_TILDE'
    else:
        return u'O'

if __name__ == '__main__':
    corpus_file = open('corpus_cess_esp.txt', 'w')
    
    for oracion in nltk.corpus.cess_esp.sents():       
        for palabra in oracion:
            palabra_norm = palabra.decode("Latin1")
            palabra_output = palabra_norm + " " + get_label(palabra_norm.lower()) + "\n"
            corpus_file.write(palabra_output.encode('utf-8'))
        corpus_file.write("\n".encode('utf-8'))
            
    corpus_file.close()