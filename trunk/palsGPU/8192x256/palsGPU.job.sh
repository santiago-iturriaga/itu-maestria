#oarsub -p "gpu='YES'" -l walltime=40:00:00 -n "palsGPU-8k" /home/clusterusers/siturriaga/itu-maestria/trunk/palsGPU/8192x256/palsGPU.8k.sh
oarsub -p "gpu='YES'" -l core=12,walltime=40:00:00 -n "palsGPU-8k" /home/clusterusers/siturriaga/itu-maestria/trunk/palsGPU/8192x256/palsGPU.shared.8k.sh
