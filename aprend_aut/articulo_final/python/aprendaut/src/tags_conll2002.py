#encoding: utf-8
'''
Created on Feb 13, 2011

@author: santiago
'''

import nltk

if __name__ == '__main__':
    for tag_pair in nltk.corpus.conll2002.tagged_words():       
        word = tag_pair[0].lower()
        tag = tag_pair[1]
        
        if word == u'cuándo' or word == u'cuánto' or word == u'dónde' or word == u'cómo' \
        or word == u'adónde' or word == u'qué' or word == u'cuando' or word == u'cuanto' \
        or word == u'donde' or word == u'como' or word == u'adonde' or word == u'que':
            print "%s/%s" % (word, tag)