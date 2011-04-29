#encoding: utf-8

'''
Created on Apr 28, 2011

@author: santiago
'''

import nltk
import sys

def palabra_sin_tilde(palabra):
    if palabra == 'cuándo':
        return 'cuando'
    elif palabra == 'cuánto':
        return 'cuanto'
    elif palabra == 'dónde':
        return 'donde'
    elif palabra == 'cómo':
        return 'como'
    elif palabra == 'adónde':
        return 'adonde'
    elif palabra == 'qué':
        return 'que'
    
    return palabra

def palabra_con_tilde(palabra):
    if palabra == 'cuando':
        return 'cuándo'
    elif palabra == 'cuanto':
        return 'cuánto'
    elif palabra == 'donde':
        return 'dónde'
    elif palabra == 'como':
        return 'cómo'
    elif palabra == 'adonde':
        return 'adónde'
    elif palabra == 'que':
        return 'qué'
    
    return palabra

def palabra_invertir_tilde(palabra):
    if palabra == 'cuándo':
        return 'cuando'
    elif palabra == 'cuánto':
        return 'cuanto'
    elif palabra == 'dónde':
        return 'donde'
    elif palabra == 'cómo':
        return 'como'
    elif palabra == 'adónde':
        return 'adonde'
    elif palabra == 'qué':
        return 'que'
    elif palabra == 'cuando':
        return 'cuándo'
    elif palabra == 'cuanto':
        return 'cuánto'
    elif palabra == 'donde':
        return 'dónde'
    elif palabra == 'como':
        return 'cómo'
    elif palabra == 'adonde':
        return 'adónde'
    elif palabra == 'que':
        return 'qué'
    
    return palabra

if __name__ == '__main__':
    O_tp = 0
    O_fp = 0
    O_fn = 0
    
    SIN_TILDE_tp = 0
    SIN_TILDE_fp = 0
    SIN_TILDE_fn = 0
    
    CON_TILDE_tp = 0
    CON_TILDE_fp = 0
    CON_TILDE_fn = 0
    
    metricas_por_adv = {}

    for i in range(9):
        reference_uri = "/home/santiago/eclipse/java-workspace/AAI/corpus/test_full_" + str(i) + ".txt"
        test_uri = "/home/santiago/eclipse/java-workspace/AAI/model_svm/f" + str(i) + "_result.txt"
        
        reference_file = open(reference_uri)
        reference = []
        for line in reference_file:
            if line.strip() != "":
                words = line.strip().split(" ")
                
                if len(words) >= 2:
                    word = words[0].strip()
                    token = words[1].strip()
                    reference.append((token, word))
            
        test_file = open(test_uri)
        test = []
        for line in test_file:
            if line.strip() != "":
                words = line.strip().split(" ")
                
                if len(words) >= 2:
                    word = words[0].strip()
                    token = words[1].strip()
                    test.append(token)
                                  
        print "[%d] Test: %d, Reference: %d" % (i, len(test), len(reference))
        assert(len(test) == len(reference))   
    
        for index in range(len(test)):
            if reference[index][0] == 'SIN_TILDE' or reference[index][0] == 'CON_TILDE':
                adverbio = reference[index][1].lower().strip()
                if not metricas_por_adv.has_key(adverbio):
                    metricas_por_adv[adverbio] = {'total': 0, 'tp': 0, 'fp': 0, 'fn': 0}
                    
                adverbio_inv = palabra_invertir_tilde(adverbio)
                if not metricas_por_adv.has_key(adverbio_inv):
                    metricas_por_adv[adverbio_inv] = {'total': 0, 'tp': 0, 'fp': 0, 'fn': 0}
            
                if reference[index][0] == 'SIN_TILDE':
                    metricas_por_adv[adverbio]['total'] = metricas_por_adv[adverbio]['total'] + 1
                elif reference[index][0] == 'CON_TILDE':
                    metricas_por_adv[adverbio_inv]['total'] = metricas_por_adv[adverbio_inv]['total'] + 1
            
            if reference[index][0] == test[index]:
                if reference[index][0] == 'O':
                    O_tp = O_tp + 1
                else:
                    palabra = reference[index][1].lower().strip()
                    
                    if reference[index][0] == 'SIN_TILDE':
                        metricas_por_adv[palabra]['tp'] = metricas_por_adv[palabra]['tp'] + 1
                        SIN_TILDE_tp = SIN_TILDE_tp + 1
                        
                    elif reference[index][0] == 'CON_TILDE':
                        palabra_tilde = palabra_con_tilde(palabra)
                        metricas_por_adv[palabra_tilde]['tp'] = metricas_por_adv[palabra_tilde]['tp'] + 1
                        CON_TILDE_tp = CON_TILDE_tp + 1
                    
            elif reference[index][0] != test[index]:
                if reference[index][0] == 'O':
                    if test[index] == 'SIN_TILDE':
                        O_fn = O_fn + 1
                        SIN_TILDE_fp = SIN_TILDE_fp + 1
                    elif test[index] == 'CON_TILDE':
                        O_fn = O_fn + 1
                        CON_TILDE_fp = CON_TILDE_fp + 1
                else:
                    if reference[index][0] == 'SIN_TILDE':
                        if test[index] == 'O':
                            SIN_TILDE_fn = SIN_TILDE_fn + 1
                            O_fp = O_fp + 1
                        elif test[index] == 'CON_TILDE':
                            palabra = reference[index][1].lower().strip()
                            metricas_por_adv[palabra]['fn'] = metricas_por_adv[palabra]['fn'] + 1
    
                            palabra_inv = palabra_invertir_tilde(reference[index][1].lower().strip())
                            metricas_por_adv[palabra_inv]['fp'] = metricas_por_adv[palabra_inv]['fp'] + 1
                            
                            SIN_TILDE_fn = SIN_TILDE_fn + 1
                            CON_TILDE_fp = CON_TILDE_fp + 1
                    elif reference[index][0] == 'CON_TILDE':
                        if test[index] == 'SIN_TILDE':
                            palabra = reference[index][1].lower().strip()
                            metricas_por_adv[palabra]['fp'] = metricas_por_adv[palabra]['fp'] + 1
                            
                            palabra_inv = palabra_invertir_tilde(reference[index][1].lower().strip())
                            metricas_por_adv[palabra_inv]['fn'] = metricas_por_adv[palabra_inv]['fn'] + 1
                            
                            CON_TILDE_fn = CON_TILDE_fn + 1
                            SIN_TILDE_fp = SIN_TILDE_fp + 1
                        if test[index] == 'O':
                            CON_TILDE_fn = CON_TILDE_fn + 1
                            O_fp = O_fp + 1
                   
    print "\n[O] tp=%d fp=%d fn=%d" % (O_tp, O_fp, O_fn)
    print "[SIN_TILDE] tp=%d fp=%d fn=%d" % (SIN_TILDE_tp, SIN_TILDE_fp, SIN_TILDE_fn)
    print "[CON_TILDE] tp=%d fp=%d fn=%d\n" % (CON_TILDE_tp, CON_TILDE_fp, CON_TILDE_fn)
    
    if O_tp > 0:
        precision_O = float(O_tp) / float(O_tp + O_fp)
        recall_O = float(O_tp) / float(O_tp + O_fn)
        f_measure_O = 2*precision_O*recall_O/(precision_O+recall_O)
    else:
        precision_O = 0.0
        recall_O = 0.0
        f_measure_O = 0.0
    print "[O] Precision: %f" % precision_O
    print "[O] Recall: %f" % recall_O
    print "[O] F-measure: %f" % f_measure_O

    if SIN_TILDE_tp > 0:
        precision_SIN_TILDE = float(SIN_TILDE_tp) / float(SIN_TILDE_tp + SIN_TILDE_fp)
        recall_SIN_TILDE = float(SIN_TILDE_tp) / float(SIN_TILDE_tp + SIN_TILDE_fn)
        f_measure_SIN_TILDE = 2*precision_SIN_TILDE*recall_SIN_TILDE/(precision_SIN_TILDE+recall_SIN_TILDE)
    else:
        precision_SIN_TILDE = 0.0
        recall_SIN_TILDE = 0.0
        f_measure_SIN_TILDE = 0.0
    print "[SIN_TILDE] Precision: %f" % precision_SIN_TILDE
    print "[SIN_TILDE] Recall: %f" % recall_SIN_TILDE
    print "[SIN_TILDE] F-measure: %f" % f_measure_SIN_TILDE

    if CON_TILDE_tp > 0:
        precision_CON_TILDE = float(CON_TILDE_tp) / float(CON_TILDE_tp + CON_TILDE_fp)
        recall_CON_TILDE = float(CON_TILDE_tp) / float(CON_TILDE_tp + CON_TILDE_fn)
        f_measure_CON_TILDE = 2*precision_CON_TILDE*recall_CON_TILDE/(precision_CON_TILDE+recall_CON_TILDE)
    else:
        precision_CON_TILDE = 0.0
        recall_CON_TILDE = 0.0
        f_measure_CON_TILDE = 0.0
    print "[CON_TILDE] Precision: %f" % precision_CON_TILDE
    print "[CON_TILDE] Recall: %f" % recall_CON_TILDE
    print "[CON_TILDE] F-measure: %f\n" % f_measure_CON_TILDE
    
    for adv in sorted(metricas_por_adv.keys()):
        if metricas_por_adv[adv]['tp'] > 0:
            precision_adv = float(metricas_por_adv[adv]['tp']) / float(metricas_por_adv[adv]['tp']+metricas_por_adv[adv]['fp'])
            recall_adv = float(metricas_por_adv[adv]['tp']) / float(metricas_por_adv[adv]['tp']+metricas_por_adv[adv]['fn'])
            f_measure_adv = 2 * precision_adv * recall_adv / (precision_adv + recall_adv)
        else:
            precision_adv = 0.0
            recall_adv = 0.0
            f_measure_adv = 0.0
        print "[%s] (total=%d) tp=%d fp=%d fn=%d" % (adv, metricas_por_adv[adv]['total'], metricas_por_adv[adv]['tp'], metricas_por_adv[adv]['fp'], metricas_por_adv[adv]['fn'])
        print "[%s] Precision: %f" % (adv, precision_adv)
        print "[%s] Recall: %f" % (adv, recall_adv)
        print "[%s] F-measure: %f\n" % (adv, f_measure_adv)