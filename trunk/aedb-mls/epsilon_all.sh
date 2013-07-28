MOEA="/home/santiago/Dropbox/Facultad/Publicaciones/EvoComnetAEDB/IJAHUC/NIDISC/comparison/data"
MOEA="/home/siturria"

: <<'END'
./epsilon-eval.py best-pf-d100-c40.pf best_pf/All3Algs100dev-c40.pf mls-eval-d100-c40/mls-eval-d100-c10 20 10 > epsilon-d100-c40.log
./epsilon-eval.py best-pf-d100-c60.pf best_pf/All3Algs100dev-c60.pf mls-eval-d100-c60/mls-eval-d100-c15 20 15 > epsilon-d100-c60.log
./epsilon-eval.py best-pf-d100-c80.pf best_pf/All3Algs100dev-c80.pf mls-eval-d100-c80/mls-eval-d100-c20 20 20 > epsilon-d100-c80.log

./epsilon-eval.py best-pf-d200-c40.pf best_pf/All3Algs200dev-c40.pf mls-eval-d200-c40/mls-eval-d200-c20 20 20 > epsilon-d200-c40.log
./epsilon-eval.py best-pf-d200-c60.pf best_pf/All3Algs200dev-c60.pf mls-eval-d200-c60/mls-eval-d200-c60 14 30 > epsilon-d200-c60.log
./epsilon-eval.py best-pf-d200-c80.pf best_pf/All3Algs200dev-c80.pf mls-eval-d200-c80/mls-eval-d200-c80 10 40 > epsilon-d200-c80.log

./epsilon-eval.py best-pf-d300-c40.pf best_pf/All3Algs300dev-c40.pf mls-eval-d300-c40/mls-eval-d300-c40-1 5 30 > epsilon-d300-c40.log
./epsilon-eval.py best-pf-d300-c60.pf best_pf/All3Algs300dev-c60.pf mls-eval-d300-c60/mls-eval-d300-c60-1 5 45 > epsilon-d300-c60.log
./epsilon-eval.py best-pf-d300-c80.pf best_pf/All3Algs300dev-c80.pf mls-eval-d300-c80/mls-eval-d300-c80-1 5 60 > epsilon-d300-c80.log

echo "epsilon-d100-c40" > epsilon-final.log
./epsilon-eval-final.py best-pf-d100-c40.pf mls-eval-d100-c40/mls-eval-d100-c10 20 10 >> epsilon-final.log
./epsilon-moea.py best-pf-d100-c40.pf best_pf/All3Algs100dev-c40.pf 10 >> epsilon-final.log
echo "epsilon-d100-c60" >> epsilon-final.log
./epsilon-eval-final.py best-pf-d100-c60.pf mls-eval-d100-c60/mls-eval-d100-c15 20 15 >> epsilon-final.log
./epsilon-moea.py best-pf-d100-c60.pf best_pf/All3Algs100dev-c60.pf 15 >> epsilon-final.log
echo "epsilon-d100-c80" >> epsilon-final.log
./epsilon-eval-final.py best-pf-d100-c80.pf mls-eval-d100-c80/mls-eval-d100-c20 20 20 >> epsilon-final.log
./epsilon-moea.py best-pf-d100-c80.pf best_pf/All3Algs100dev-c80.pf 20 >> epsilon-final.log

echo "epsilon-d200-c40" >> epsilon-final.log
./epsilon-eval-final.py best-pf-d200-c40.pf mls-eval-d200-c40/mls-eval-d200-c20 20 20 >> epsilon-final.log
./epsilon-moea.py best-pf-d200-c40.pf best_pf/All3Algs200dev-c40.pf 20 >> epsilon-final.log
echo "epsilon-d200-c60" >> epsilon-final.log
./epsilon-eval-final.py best-pf-d200-c60.pf mls-eval-d200-c60/mls-eval-d200-c60 14 30 >> epsilon-final.log
./epsilon-moea.py best-pf-d200-c60.pf best_pf/All3Algs200dev-c60.pf 30 >> epsilon-final.log
echo "epsilon-d200-c80" >> epsilon-final.log
./epsilon-eval-final.py best-pf-d200-c80.pf mls-eval-d200-c80/mls-eval-d200-c80 10 40 >> epsilon-final.log
./epsilon-moea.py best-pf-d200-c80.pf best_pf/All3Algs200dev-c80.pf 40 >> epsilon-final.log

echo "epsilon-d300-c40" >> epsilon-final.log
./epsilon-eval-final.py best-pf-d300-c40.pf mls-eval-d300-c40/mls-eval-d300-c40-1 5 30 >> epsilon-final.log
./epsilon-moea.py best-pf-d300-c40.pf best_pf/All3Algs300dev-c40.pf 30 >> epsilon-final.log
echo "epsilon-d300-c60" >> epsilon-final.log
./epsilon-eval-final.py best-pf-d300-c60.pf mls-eval-d300-c60/mls-eval-d300-c60-1 5 45 >> epsilon-final.log
./epsilon-moea.py best-pf-d300-c60.pf best_pf/All3Algs300dev-c60.pf 45 >> epsilon-final.log
echo "epsilon-d100-c80" >> epsilon-final.log
./epsilon-eval-final.py best-pf-d300-c80.pf mls-eval-d300-c80/mls-eval-d300-c80-1 5 60 >> epsilon-final.log
./epsilon-moea.py best-pf-d300-c80.pf best_pf/All3Algs300dev-c80.pf 60 >> epsilon-final.log
END

echo "epsilon-d100-c40"
./epsilon-eval-final.py best-pf-d100-c40.pf mls-eval-d100-c40/mls-eval-d100-c10 20 10 > mls_epsilon-d100-c40.log
./epsilon-eval-moea.py best-pf-d100-c40.pf ${MOEA}/CellDE/100dev/FUN 30 10 > cellde_epsilon-d100-c40.log
./epsilon-eval-moea.py best-pf-d100-c40.pf ${MOEA}/NSGAII/100dev/FUN 30 10 > nsgaii_epsilon-d100-c40.log

echo "epsilon-d100-c60"
./epsilon-eval-final.py best-pf-d100-c60.pf mls-eval-d100-c60/mls-eval-d100-c15 20 15 > mls_epsilon-d100-c60.log
./epsilon-eval-moea.py best-pf-d100-c60.pf ${MOEA}/CellDE/100dev/FUN 30 15 > cellde_epsilon-d100-c60.log
./epsilon-eval-moea.py best-pf-d100-c60.pf ${MOEA}/NSGAII/100dev/FUN 30 15 > nsgaii_epsilon-d100-c60.log
echo "epsilon-d100-c80"
./epsilon-eval-final.py best-pf-d100-c80.pf mls-eval-d100-c80/mls-eval-d100-c20 20 20 > mls_epsilon-d100-c80.log
./epsilon-eval-moea.py best-pf-d100-c80.pf ${MOEA}/CellDE/100dev/FUN 30 20 > cellde_epsilon-d100-c80.log
./epsilon-eval-moea.py best-pf-d100-c80.pf ${MOEA}/NSGAII/100dev/FUN 30 20 > nsgaii_epsilon-d100-c80.log

echo "epsilon-d200-c40"
./epsilon-eval-final.py best-pf-d200-c40.pf mls-eval-d200-c40/mls-eval-d200-c20 20 20 > mls_epsilon-d200-c40.log
./epsilon-eval-moea.py best-pf-d200-c40.pf ${MOEA}/CellDE/200dev/FUN 30 20 > cellde_epsilon-d200-c40.log
./epsilon-eval-moea.py best-pf-d200-c40.pf ${MOEA}/NSGAII/200dev/FUN 30 20 > nsgaii_epsilon-d200-c40.log
echo "epsilon-d200-c60"
./epsilon-eval-final.py best-pf-d200-c60.pf mls-eval-d200-c60/mls-eval-d200-c60 20 30 > mls_epsilon-d200-c60.log
./epsilon-eval-moea.py best-pf-d200-c60.pf ${MOEA}/CellDE/200dev/FUN 30 30 > cellde_epsilon-d200-c60.log
./epsilon-eval-moea.py best-pf-d200-c60.pf ${MOEA}/NSGAII/200dev/FUN 30 10 > nsgaii_epsilon-d200-c60.log
echo "epsilon-d200-c80"
./epsilon-eval-final.py best-pf-d200-c80.pf mls-eval-d200-c80/mls-eval-d200-c80 20 40 > mls_epsilon-d200-c80.log
./epsilon-eval-moea.py best-pf-d200-c80.pf ${MOEA}/CellDE/200dev/FUN 30 40 > cellde_epsilon-d200-c80.log
./epsilon-eval-moea.py best-pf-d200-c80.pf ${MOEA}/NSGAII/200dev/FUN 30 40 > nsgaii_epsilon-d200-c80.log

echo "epsilon-d300-c40"
./epsilon-eval-final.py best-pf-d300-c40.pf mls-eval-d300-c40/mls-eval-d300-c40-1 10 30 > mls_epsilon-d300-c40.log
./epsilon-eval-moea.py best-pf-d300-c40.pf ${MOEA}/CellDE/300dev/FUN 30 30 > cellde_epsilon-d300-c40.log
./epsilon-eval-moea.py best-pf-d300-c40.pf ${MOEA}/NSGAII/300dev/FUN 30 30 > nsgaii_epsilon-d300-c40.log
echo "epsilon-d300-c60"
./epsilon-eval-final.py best-pf-d300-c60.pf mls-eval-d300-c60/mls-eval-d300-c60-1 8 45 > mls_epsilon-d300-c60.log
./epsilon-eval-moea.py best-pf-d300-c60.pf ${MOEA}/CellDE/300dev/FUN 30 45 > cellde_epsilon-d300-c60.log
./epsilon-eval-moea.py best-pf-d300-c60.pf ${MOEA}/NSGAII/300dev/FUN 30 45 > nsgaii_epsilon-d300-c60.log
echo "epsilon-d300-c80"
./epsilon-eval-final.py best-pf-d300-c80.pf mls-eval-d300-c80/mls-eval-d300-c80-1 5 60 > mls_epsilon-d300-c80.log
./epsilon-eval-moea.py best-pf-d300-c80.pf ${MOEA}/CellDE/300dev/FUN 30 60 > cellde_epsilon-d300-c80.log
./epsilon-eval-moea.py best-pf-d300-c80.pf ${MOEA}/NSGAII/300dev/FUN 30 60 > nsgaii_epsilon-d300-c80.log
