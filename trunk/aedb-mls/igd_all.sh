MOEA="/home/santiago/Dropbox/Facultad/Publicaciones/EvoComnetAEDB/IJAHUC/NIDISC/comparison/data"

./igd-eval.py best-pf-d100-c40.pf best_pf/All3Algs100dev-c40.pf mls-eval-d100-c40/mls-eval-d100-c10 20 10 > igd-d100-c40.log
./igd-eval.py best-pf-d100-c60.pf best_pf/All3Algs100dev-c60.pf mls-eval-d100-c60/mls-eval-d100-c15 20 15 > igd-d100-c60.log
./igd-eval.py best-pf-d100-c80.pf best_pf/All3Algs100dev-c80.pf mls-eval-d100-c80/mls-eval-d100-c20 20 20 > igd-d100-c80.log

./igd-eval.py best-pf-d200-c40.pf best_pf/All3Algs200dev-c40.pf mls-eval-d200-c40/mls-eval-d200-c20 20 20 > igd-d200-c40.log
./igd-eval.py best-pf-d200-c60.pf best_pf/All3Algs200dev-c60.pf mls-eval-d200-c60/mls-eval-d200-c60 20 30 > igd-d200-c60.log
./igd-eval.py best-pf-d200-c80.pf best_pf/All3Algs200dev-c80.pf mls-eval-d200-c80/mls-eval-d200-c80 20 40 > igd-d200-c80.log

./igd-eval.py best-pf-d300-c40.pf best_pf/All3Algs300dev-c40.pf mls-eval-d300-c40/mls-eval-d300-c40-1 10 30 > igd-d300-c40.log
./igd-eval.py best-pf-d300-c60.pf best_pf/All3Algs300dev-c60.pf mls-eval-d300-c60/mls-eval-d300-c60-1 8 45 > igd-d300-c60.log
./igd-eval.py best-pf-d300-c80.pf best_pf/All3Algs300dev-c80.pf mls-eval-d300-c80/mls-eval-d300-c80-1 5 60 > igd-d300-c80.log

#echo "igd-d100-c40" > igd-final.log
#./igd-eval-final.py best-pf-d100-c40.pf mls-eval-d100-c40/mls-eval-d100-c10 20 10 >> igd-final.log
#./igd-moea.py best-pf-d100-c40.pf best_pf/All3Algs100dev-c40.pf 10 >> igd-final.log
#echo "igd-d100-c60" >> igd-final.log
#./igd-eval-final.py best-pf-d100-c60.pf mls-eval-d100-c60/mls-eval-d100-c15 20 15 >> igd-final.log
#./igd-moea.py best-pf-d100-c60.pf best_pf/All3Algs100dev-c60.pf 15 >> igd-final.log
#echo "igd-d100-c80" >> igd-final.log
#./igd-eval-final.py best-pf-d100-c80.pf mls-eval-d100-c80/mls-eval-d100-c20 20 20 >> igd-final.log
#./igd-moea.py best-pf-d100-c80.pf best_pf/All3Algs100dev-c80.pf 20 >> igd-final.log

#echo "igd-d200-c40" >> igd-final.log
#./igd-eval-final.py best-pf-d200-c40.pf mls-eval-d200-c40/mls-eval-d200-c20 20 20 >> igd-final.log
#./igd-moea.py best-pf-d200-c40.pf best_pf/All3Algs200dev-c40.pf 20 >> igd-final.log
#echo "igd-d200-c60" >> igd-final.log
#./igd-eval-final.py best-pf-d200-c60.pf mls-eval-d200-c60/mls-eval-d200-c60 14 30 >> igd-final.log
#./igd-moea.py best-pf-d200-c60.pf best_pf/All3Algs200dev-c60.pf 30 >> igd-final.log
#echo "igd-d200-c80" >> igd-final.log
#./igd-eval-final.py best-pf-d200-c80.pf mls-eval-d200-c80/mls-eval-d200-c80 10 40 >> igd-final.log
#./igd-moea.py best-pf-d200-c80.pf best_pf/All3Algs200dev-c80.pf 40 >> igd-final.log

#echo "igd-d300-c40" >> igd-final.log
#./igd-eval-final.py best-pf-d300-c40.pf mls-eval-d300-c40/mls-eval-d300-c40-1 5 30 >> igd-final.log
#./igd-moea.py best-pf-d300-c40.pf best_pf/All3Algs300dev-c40.pf 30 >> igd-final.log
#echo "igd-d300-c60" >> igd-final.log
#./igd-eval-final.py best-pf-d300-c60.pf mls-eval-d300-c60/mls-eval-d300-c60-1 5 45 >> igd-final.log
#./igd-moea.py best-pf-d300-c60.pf best_pf/All3Algs300dev-c60.pf 45 >> igd-final.log
#echo "igd-d300-c80" >> igd-final.log
#./igd-eval-final.py best-pf-d300-c80.pf mls-eval-d300-c80/mls-eval-d300-c80-1 5 60 >> igd-final.log
#./igd-moea.py best-pf-d300-c80.pf best_pf/All3Algs300dev-c80.pf 60 >> igd-final.log

echo "igd-d100-c40"
./igd-eval-final.py best-pf-d100-c40.pf mls-eval-d100-c40/mls-eval-d100-c10 20 10 > mls_igd-d100-c40.log
./igd-eval-moea.py best-pf-d100-c40.pf ${MOEA}/CellDE/100dev/FUN 30 10 > cellde_igd-d100-c40.log
./igd-eval-moea.py best-pf-d100-c40.pf ${MOEA}/NSGAII/100dev/FUN 30 10 > nsgaii_igd-d100-c40.log
echo "igd-d100-c60"
./igd-eval-final.py best-pf-d100-c60.pf mls-eval-d100-c60/mls-eval-d100-c15 20 15 > mls_igd-d100-c60.log
./igd-eval-moea.py best-pf-d100-c60.pf ${MOEA}/CellDE/100dev/FUN 30 15 > cellde_igd-d100-c60.log
./igd-eval-moea.py best-pf-d100-c60.pf ${MOEA}/NSGAII/100dev/FUN 30 15 > nsgaii_igd-d100-c60.log
echo "igd-d100-c80"
./igd-eval-final.py best-pf-d100-c80.pf mls-eval-d100-c80/mls-eval-d100-c20 20 20 > mls_igd-d100-c80.log
./igd-eval-moea.py best-pf-d100-c80.pf ${MOEA}/CellDE/100dev/FUN 30 20 > cellde_igd-d100-c80.log
./igd-eval-moea.py best-pf-d100-c80.pf ${MOEA}/NSGAII/100dev/FUN 30 20 > nsgaii_igd-d100-c80.log

echo "igd-d200-c40"
./igd-eval-final.py best-pf-d200-c40.pf mls-eval-d200-c40/mls-eval-d200-c20 20 20 > mls_igd-d200-c40.log
./igd-eval-moea.py best-pf-d200-c40.pf ${MOEA}/CellDE/200dev/FUN 30 20 > cellde_igd-d200-c40.log
./igd-eval-moea.py best-pf-d200-c40.pf ${MOEA}/NSGAII/200dev/FUN 30 20 > nsgaii_igd-d200-c40.log
echo "igd-d200-c60"
./igd-eval-final.py best-pf-d200-c60.pf mls-eval-d200-c60/mls-eval-d200-c60 20 30 > mls_igd-d200-c60.log
./igd-eval-moea.py best-pf-d200-c60.pf ${MOEA}/CellDE/200dev/FUN 30 30 > cellde_igd-d200-c60.log
./igd-eval-moea.py best-pf-d200-c60.pf ${MOEA}/NSGAII/200dev/FUN 30 10 > nsgaii_igd-d200-c60.log
echo "igd-d200-c80"
./igd-eval-final.py best-pf-d200-c80.pf mls-eval-d200-c80/mls-eval-d200-c80 20 40 > mls_igd-d200-c80.log
./igd-eval-moea.py best-pf-d200-c80.pf ${MOEA}/CellDE/200dev/FUN 30 40 > cellde_igd-d200-c80.log
./igd-eval-moea.py best-pf-d200-c80.pf ${MOEA}/NSGAII/200dev/FUN 30 40 > nsgaii_igd-d200-c80.log

echo "igd-d300-c40"
./igd-eval-final.py best-pf-d300-c40.pf mls-eval-d300-c40/mls-eval-d300-c40-1 10 30 > mls_igd-d300-c40.log
./igd-eval-moea.py best-pf-d200-c40.pf ${MOEA}/CellDE/300dev/FUN 30 30 > cellde_igd-d200-c40.log
./igd-eval-moea.py best-pf-d200-c40.pf ${MOEA}/NSGAII/300dev/FUN 30 30 > nsgaii_igd-d200-c40.log
echo "igd-d300-c60"
./igd-eval-final.py best-pf-d300-c60.pf mls-eval-d300-c60/mls-eval-d300-c60-1 8 45 > mls_igd-d300-c60.log
./igd-eval-moea.py best-pf-d200-c60.pf ${MOEA}/CellDE/300dev/FUN 30 45 > cellde_igd-d200-c60.log
./igd-eval-moea.py best-pf-d200-c60.pf ${MOEA}/NSGAII/300dev/FUN 30 45 > nsgaii_igd-d200-c60.log
echo "igd-d300-c80"
./igd-eval-final.py best-pf-d300-c80.pf mls-eval-d300-c80/mls-eval-d300-c80-1 5 60 > mls_igd-d300-c80.log
./igd-eval-moea.py best-pf-d100-c80.pf ${MOEA}/CellDE/300dev/FUN 30 60 > cellde_igd-d100-c80.log
./igd-eval-moea.py best-pf-d100-c80.pf ${MOEA}/NSGAII/300dev/FUN 30 60 > nsgaii_igd-d100-c80.log
