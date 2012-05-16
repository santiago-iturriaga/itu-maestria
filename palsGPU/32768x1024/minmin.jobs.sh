oarsub -l walltime=20:00:00 -n "minmin-1" /home/clusterusers/siturriaga/itu-maestria/trunk/palsGPU/32768x1024/minmin.32k.1.sh
sleep 1
oarsub -l walltime=20:00:00 -n "minmin-2" /home/clusterusers/siturriaga/itu-maestria/trunk/palsGPU/32768x1024/minmin.32k.2.sh

