#encoding: utf-8

import sys
import math

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Uso: %s <file>" % sys.argv[0]
        print "Error!!!"
        print sys.argv
        exit();
    
    is_first = True
        
    file_path = sys.argv[1]
    workload = open(file_path)
    
    for line in workload:
        if is_first:
            is_first = False
        else:
            print line,
