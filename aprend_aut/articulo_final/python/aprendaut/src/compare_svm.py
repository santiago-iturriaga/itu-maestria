#coding: utf-8
'''
Created on Mar 30, 2011

@author: santiago
'''

if __name__ == '__main__':
    ok = 0
    error = 0
    CON_TILDE_ok = 0
    CON_TILDE_error = 0
    SIN_TILDE_ok = 0
    SIN_TILDE_error = 0
    
    test_file = open('SVM_Test_Final-full.log','r')
    
    for output_line in open('SVM_Output.log', 'r'):
        test_line = test_file.readline()
        
        test_line_partes = test_line.split(" ")
        if test_line_partes[0].strip() == output_line.strip():
            ok = ok + 1
            print "OK!\n"
            
            if output_line.strip() == '2':
                SIN_TILDE_ok = SIN_TILDE_ok + 1
            elif output_line.strip() == '3':
                CON_TILDE_ok = CON_TILDE_ok + 1
        else:
            error = error + 1 
            print "Debería ser %s pero es %s\n" % (test_line_partes[0].strip(), output_line.strip())

            if test_line_partes[0].strip() == '2':
                SIN_TILDE_error = SIN_TILDE_error + 1
            elif test_line_partes[0].strip() == '3':
                CON_TILDE_error = CON_TILDE_error + 1
                
    print "CON_TILDE\n"
    print "   ok   = %s\n" % CON_TILDE_ok
    print "  error = %s\n" % CON_TILDE_error
    print "SIN_TILDE\n"
    print "   ok   = %s\n" % SIN_TILDE_ok
    print "  error = %s\n" % SIN_TILDE_error
                