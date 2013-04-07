#encoding: utf-8

import sys
import os
import math
import numpy

directory = 'prng_3'

cant_iters = 15
threads_list = (1,2,4,6,8,10,12,14,16,18,20,22,24)
prng_list = ('randr','drand48r','mt')

if __name__ == '__main__':  
    resultados_pals_info = {}
    
    for prng in prng_list:
        resultados_pals_info[prng] = {}
        
        for threads in threads_list:
            resultados_pals_info[prng][threads] = []
            
            for i in range(cant_iters):
                f = directory + '/gmon.' + prng + '.' + str(threads) + '.' + str(i) + '.info'

                if os.path.isfile(f):
                    info_f = open(f)

                    for line in info_f:
                        values = line.strip().split('|')

                        if values[0] == 'TOTAL_TIME':
                            resultados_pals_info[prng][threads].append(float(values[1]) / 1000000.0)

                else:
                    print "[ERROR] cargando info de la heuristica pals"

    #print resultados_pals_info
    #np_res = numpy.array(resultados_pals_info)
    
    print " & \\multicolumn{" + str(len(prng_list)) +"}{c}{time (s) \\\\"
    
    print "\\multicolumn{1}{c}{\\textbf{threads}}",
    for prng in prng_list:
        print "& \\multicolumn{1}{c}{\\textbf{" + prng + "}}",
    print "\\\\"

    for threads in threads_list:
        print "$%d$" % (threads),
        
        for prng in prng_list:
            aux = numpy.array(resultados_pals_info[prng][threads])
            print "& $%.1f\pm%.2f$" % (aux.mean(), aux.std()),
            
        print "\\\\"

    print "==========================================="
    
    print "threads",
    for prng in prng_list:
        print "," + prng,
    print ""

    for threads in threads_list:
        print "%d" % (threads),
        
        for prng in prng_list:
            aux = numpy.array(resultados_pals_info[prng][threads])
            print ",%.1f" % (aux.mean()),
            
        print ""
