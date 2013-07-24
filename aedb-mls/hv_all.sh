./hypervolume-eval.py best-pf-d100-c40.pf best_pf/All3Algs100dev-c40.pf mls-eval-d100-c40/mls-eval-d100-c10 20 10 > hv-d100-c40.log
./hypervolume-eval.py best-pf-d100-c60.pf best_pf/All3Algs100dev-c60.pf mls-eval-d100-c60/mls-eval-d100-c15 20 15 > hv-d100-c60.log
./hypervolume-eval.py best-pf-d100-c80.pf best_pf/All3Algs100dev-c80.pf mls-eval-d100-c80/mls-eval-d100-c20 20 20 > hv-d100-c80.log

./hypervolume-eval.py best-pf-d200-c40.pf best_pf/All3Algs200dev-c40.pf mls-eval-d200-c40/mls-eval-d200-c20 20 20 > hv-d200-c40.log
./hypervolume-eval.py best-pf-d200-c60.pf best_pf/All3Algs200dev-c60.pf mls-eval-d200-c60/mls-eval-d200-c60 14 30 > hv-d200-c60.log
./hypervolume-eval.py best-pf-d200-c80.pf best_pf/All3Algs200dev-c80.pf mls-eval-d200-c80/mls-eval-d200-c80 10 40 > hv-d200-c80.log

./hypervolume-eval.py best-pf-d300-c40.pf best_pf/All3Algs300dev-c40.pf mls-eval-d300-c40/mls-eval-d300-c40-1 5 30 > hv-d300-c40.log
./hypervolume-eval.py best-pf-d300-c60.pf best_pf/All3Algs300dev-c60.pf mls-eval-d300-c60/mls-eval-d300-c60-1 5 45 > hv-d300-c60.log
./hypervolume-eval.py best-pf-d300-c80.pf best_pf/All3Algs300dev-c80.pf mls-eval-d300-c80/mls-eval-d300-c80-1 5 60 > hv-d300-c80.log

echo "hv-d100-c40" > hv-final.log
./hypervolume-eval-final.py best-pf-d100-c40.pf mls-eval-d100-c40/mls-eval-d100-c10 20 10 >> hv-final.log
./hypervolume-moea.py best-pf-d100-c40.pf best_pf/All3Algs100dev-c40.pf 10 >> hv-final.log
echo "hv-d100-c60" >> hv-final.log
./hypervolume-eval-final.py best-pf-d100-c60.pf mls-eval-d100-c60/mls-eval-d100-c15 20 15 >> hv-final.log
./hypervolume-moea.py best-pf-d100-c60.pf best_pf/All3Algs100dev-c60.pf 15 >> hv-final.log
echo "hv-d100-c80" >> hv-final.log
./hypervolume-eval-final.py best-pf-d100-c80.pf mls-eval-d100-c80/mls-eval-d100-c20 20 20 >> hv-final.log
./hypervolume-moea.py best-pf-d100-c80.pf best_pf/All3Algs100dev-c80.pf 20 >> hv-final.log

echo "hv-d200-c40" >> hv-final.log
./hypervolume-eval-final.py best-pf-d200-c40.pf mls-eval-d200-c40/mls-eval-d200-c20 20 20 >> hv-final.log
./hypervolume-moea.py best-pf-d200-c40.pf best_pf/All3Algs200dev-c40.pf 20 >> hv-final.log
echo "hv-d200-c60" >> hv-final.log
./hypervolume-eval-final.py best-pf-d200-c60.pf mls-eval-d200-c60/mls-eval-d200-c60 14 30 >> hv-final.log
./hypervolume-moea.py best-pf-d200-c60.pf best_pf/All3Algs200dev-c60.pf 30 >> hv-final.log
echo "hv-d200-c80" >> hv-final.log
./hypervolume-eval-final.py best-pf-d200-c80.pf mls-eval-d200-c80/mls-eval-d200-c80 10 40 >> hv-final.log
./hypervolume-moea.py best-pf-d200-c80.pf best_pf/All3Algs200dev-c80.pf 40 >> hv-final.log

echo "hv-d300-c40" >> hv-final.log
./hypervolume-eval-final.py best-pf-d300-c40.pf mls-eval-d300-c40/mls-eval-d300-c40-1 5 30 >> hv-final.log
./hypervolume-moea.py best-pf-d300-c40.pf best_pf/All3Algs300dev-c40.pf 30 >> hv-final.log
echo "hv-d300-c60" >> hv-final.log
./hypervolume-eval-final.py best-pf-d300-c60.pf mls-eval-d300-c60/mls-eval-d300-c60-1 5 45 >> hv-final.log
./hypervolume-moea.py best-pf-d300-c60.pf best_pf/All3Algs300dev-c60.pf 45 >> hv-final.log
echo "hv-d100-c80" >> hv-final.log
./hypervolume-eval-final.py best-pf-d300-c80.pf mls-eval-d300-c80/mls-eval-d300-c80-1 5 60 >> hv-final.log
./hypervolume-moea.py best-pf-d300-c80.pf best_pf/All3Algs300dev-c80.pf 60 >> hv-final.log
