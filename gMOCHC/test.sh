cd /home/clusterusers/siturriaga/itu-maestria/trunk/gMOCHC
make clean
make cpu_r

SEED=0
THREADS=12
TIMEOUT=1
ITERATIONS=2500
time(bin/gmochc_cpu ~/instances/1024x32.ME/scenario.0 ~/instances/1024x32.ME/workload.A.u_c_hihi \
  1024 32 3 ${THREADS} ${SEED} ${TIMEOUT} ${ITERATIONS} \
  1>chc__0_A.u_c_hihi_1k.sol 2>chc__0_A.u_c_hihi_1k.log)

