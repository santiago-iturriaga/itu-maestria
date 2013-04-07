ITERATIONS=30

SCENARIOS=(0,3,6,9,10,11,13,14,16,17,19)
WORKLOADS=("A.u_c_hihi","A.u_c_hilo","A.u_c_lohi","A.u_c_lolo","B.u_c_hihi","B.u_c_hilo","B.u_c_lohi","B.u_c_lolo")

BASE_PATH_1="../../512x16.24_10s"
ALGORITHM_1="pals-aga"
METRICS_PATH_1="metrics-aga"

BASE_PATH_2="../../512x16.24.adhoc"
ALGORITHM_2="pals-1"
METRICS_PATH_2="metrics-adhoc"

BASE_PATH_fp="../../512x16.fp"

if __name__ == '__main__':
    for s in SCENARIOS:
        for w in WORKLOADS:
            print "=== scenario." + str(s) + ".workload." + w
            
            metrics_fp_path = BASE_PATH_fp + "/scenario." + str(s) + ".workload." + w + ".fp"
            #print metrics_fp_path
            
            values_fp = []
            f_fp = open(metrics_fp_path)
            for v in f_fp:
                v_aux = v.split(' ')
                values_fp.append((float(v_aux[0]), float(v_aux[1])))
            
            cant_1 = 0
            cant_2 = 0
            
            for i in range(ITERATIONS):
                metrics_1_path = BASE_PATH_1 + "/scenario." + str(s) + ".workload." + w + "." + str(i) + "/" + ALGORITHM_1 + ".scenario." + str(s) + ".workload." + w + ".metrics"
                #print metrics_1_path
                               
                values_1 = []
                f_1 = open(metrics_1_path)
                for v in f_1:
                    v_aux = v.split(' ')
                    values_1.append((float(v_aux[0]), float(v_aux[1])))
                    
                for v in values_1:
                    if v in values_fp:
                        cant_1 = cant_1 + 1
                    
                #print values_fp
                #print values_1
                #print cant_1
                
                metrics_2_path = BASE_PATH_2 + "/scenario." + str(s) + ".workload." + w + "." + str(i) + "/" + ALGORITHM_2 + ".scenario." + str(s) + ".workload." + w + ".metrics"
                #print metrics_2_path

                values_2 = []
                f_2 = open(metrics_2_path)
                for v in f_2:
                    v_aux = v.split(' ')
                    values_2.append((float(v_aux[0]), float(v_aux[1])))
                    
                for v in values_2:
                    if v in values_fp:
                        cant_2 = cant_2 + 1
                        
            #print "aga   : %s" % (cant_1)
            #print "ad hoc: %s" % (cant_2)

            print "aga   : %s" % (float(cant_1)/float(ITERATIONS))
            print "ad hoc: %s" % (float(cant_2)/float(ITERATIONS))
