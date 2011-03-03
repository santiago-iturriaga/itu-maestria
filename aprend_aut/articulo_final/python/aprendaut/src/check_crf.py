'''
Created on Feb 15, 2011

@author: santiago
'''

import sys

if __name__ == '__main__':
    file_crf = open("salida.txt","r")
    file_check = open("esp.testa-classify-check.txt","r")
    file_orig = open("esp.testa-classify-check-full.txt","r")
    
    num_total_con_tilde = 0
    num_total_sin_tilde = 0
    
    num_fail = 0
    num_ok = 0
    line_pos = 0
    
    line_crf = file_crf.readline()
    
    while len(line_crf) > 0:
        line_pos = line_pos + 1
        line_check = file_check.readline()
        line_orig = file_orig.readline()
        
        if len(line_check) > 0 and len(line_orig) > 0:
            result_crf = line_crf.strip()
            result_check = line_check.strip() 
            
            if result_check == 'CON_TILDE': num_total_con_tilde = num_total_con_tilde + 1
            if result_check == 'SIN_TILDE': num_total_sin_tilde = num_total_sin_tilde + 1
            
            result = result_crf == result_check
            if not result:
                print "[ERROR] Line %s '%s' is %s should be %s\n" % (line_pos, line_orig.strip(), result_crf, result_check)
                
                if not ((result_check == 'SIN_TILDE' and result_crf == 'O') \
                or (result_check == 'O')):
                    num_fail = num_fail + 1
                    
        line_crf = file_crf.readline()

    print "[INFO] Total SIN_TILDE: %s\n" % num_total_sin_tilde
    print "[INFO] Total CON_TILDE: %s\n\n" % num_total_con_tilde
                
    print "[INFO] Total FAIL: %s\n" % num_fail
        