#oarsub -l /nodes=1/core=12,walltime=20:00:00 -n "aedb-mls-test" /home/clusterusers/siturriaga/itu-maestria/trunk/aedb-mls/test-mls.sh
oarsub -l /nodes=2,walltime=1:00:00 -n "test-mls" /home/clusterusers/siturriaga/itu-maestria/trunk/aedb-mls/test-mls.sh
